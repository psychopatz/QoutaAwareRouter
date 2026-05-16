from typing import Any, AsyncIterator, Dict, Optional

import httpx

from ..base import BaseProvider
from ..openai_compatible import (
	ProviderCapabilities,
	build_openai_chat_payload,
	ensure_supported_request,
	send_streaming_json_request,
)
from ...errors import ErrorType, GatewayError
from ...schemas import ChatCompletionRequest, ChatCompletionResponse
from ...streaming.control import ProviderStreamControl
from .models import GeminiModelMixin
from .native import GeminiNativeMixin


class GeminiProvider(GeminiNativeMixin, GeminiModelMixin, BaseProvider):
	def __init__(self, **kwargs):
		kwargs.setdefault("supports_streaming", True)
		super().__init__(**kwargs)
		self.base_url = self.config.get(
			"base_url", "https://generativelanguage.googleapis.com/v1beta/openai"
		).rstrip("/")
		self.native_base_url = self.config.get(
			"native_base_url", "https://generativelanguage.googleapis.com/v1beta"
		).rstrip("/")
		self._models_cache = {"timestamp": 0.0, "models": []}

	@property
	def api_key(self) -> str:
		return self.config.get("api_key", "")

	@property
	def headers(self) -> Dict[str, str]:
		return {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}

	@property
	def native_headers(self) -> Dict[str, str]:
		return {
			"x-goog-api-key": self.api_key,
			"Content-Type": "application/json",
		}

	def default_capabilities(self) -> ProviderCapabilities:
		return ProviderCapabilities(
			supports_tools=True,
			supports_tool_choice=True,
			supports_parallel_tool_calls=True,
			supports_vision_input=True,
			supports_audio_input=True,
			supports_audio_output=True,
			supports_reasoning=True,
			supports_response_format=True,
		)

	def convert_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
		self._ensure_api_key()
		ensure_supported_request(request, self._capabilities_for_model_hint(request.model), f"{self.id}/{request.model}")
		if self._should_use_native_api(request):
			return self._native_payload(request)
		return build_openai_chat_payload(request)

	def convert_response(self, provider_response: Dict[str, Any], model: str) -> ChatCompletionResponse:
		response_payload = dict(provider_response)
		response_payload["model"] = model
		response_payload["provider"] = {
			"id": self.id,
			"type": self.type,
			"actual_model": provider_response.get("model", model),
		}
		return ChatCompletionResponse(**response_payload)

	def convert_stream_chunk(self, provider_chunk: bytes) -> bytes:
		return provider_chunk

	async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
		if self._should_use_native_api(request):
			return await self._chat_completion_native(request)

		async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
			response = await client.post(
				f"{self.base_url}/chat/completions",
				headers=self.headers,
				json=self.convert_request(request),
			)
			if response.status_code != 200:
				raise self.normalize_error(response)
			return self.convert_response(response.json(), request.model)

	async def _chat_completion_native(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
		async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
			response = await client.post(
				f"{self.native_base_url}/{self._native_model_name(request.model)}:generateContent",
				headers=self.native_headers,
				json=self._native_payload(request),
			)
			if response.status_code != 200:
				raise self.normalize_error(response)
			return self._native_to_chat_response(response.json(), request.model)

	async def stream_chat_completion(
		self,
		request: ChatCompletionRequest,
		stream_control: Optional[ProviderStreamControl] = None,
	) -> AsyncIterator[bytes]:
		if self._should_use_native_api(request):
			async for chunk in self._stream_chat_completion_native(request, stream_control=stream_control):
				yield chunk
			return

		stream_request = request.model_copy(update={"stream": True})
		client = httpx.AsyncClient(timeout=self.timeout_seconds)
		response = None

		async def abort_stream():
			if response is not None:
				await response.aclose()
			await client.aclose()

		try:
			response = await send_streaming_json_request(
				client,
				url=f"{self.base_url}/chat/completions",
				headers=self.headers,
				json_body=self.convert_request(stream_request),
			)

			if stream_control is not None:
				stream_control.register_cancel_callback(abort_stream, native_supported=True)

			if response.status_code != 200:
				await response.aread()
				raise self.normalize_error(response)

			async for chunk in response.aiter_bytes():
				yield self.convert_stream_chunk(chunk)
		finally:
			if response is not None:
				await response.aclose()
			await client.aclose()

	async def _stream_chat_completion_native(
		self,
		request: ChatCompletionRequest,
		stream_control: Optional[ProviderStreamControl] = None,
	) -> AsyncIterator[bytes]:
		client = httpx.AsyncClient(timeout=self.timeout_seconds)
		response = None
		state: Dict[str, Any] = {}

		async def abort_stream():
			if response is not None:
				await response.aclose()
			await client.aclose()

		try:
			response = await send_streaming_json_request(
				client,
				url=f"{self.native_base_url}/{self._native_model_name(request.model)}:streamGenerateContent?alt=sse",
				headers=self.native_headers,
				json_body=self._native_payload(request),
			)

			if stream_control is not None:
				stream_control.register_cancel_callback(abort_stream, native_supported=True)

			if response.status_code != 200:
				await response.aread()
				raise self.normalize_error(response)

			async for line in response.aiter_lines():
				if not line or not line.startswith("data:"):
					continue
				payload = line[5:].strip()
				if not payload:
					continue
				chunk_payload = __import__("json").loads(payload)
				for converted_chunk in self._native_stream_chunk_to_openai(chunk_payload, request.model, state):
					yield converted_chunk

			yield b"data: [DONE]\n\n"
		finally:
			if response is not None:
				await response.aclose()
			await client.aclose()

	def normalize_error(self, error: Any) -> GatewayError:
		if isinstance(error, GatewayError):
			return error

		if isinstance(error, httpx.Response):
			status_code = error.status_code
			message = error.text
			try:
				payload = error.json()
				error_payload = payload.get("error") or {}
				message = error_payload.get("message") or payload.get("message") or message
			except Exception:
				pass
		else:
			status_code = getattr(error, "status_code", 500)
			message = str(error)

		if status_code in {401, 403}:
			return GatewayError(
				message,
				type=ErrorType.AUTH_FAILED,
				provider_id=self.id,
				status_code=status_code,
			)

		if self._is_quota_error(status_code, message):
			return GatewayError(
				message,
				type=ErrorType.RATE_LIMITED,
				provider_id=self.id,
				status_code=status_code,
			)

		return GatewayError(message, provider_id=self.id, status_code=status_code)