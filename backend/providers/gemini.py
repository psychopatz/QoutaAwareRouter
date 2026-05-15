import time
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from .base import BaseProvider, ProviderHealth, ProviderModel
from .openai_compatible import (
	ProviderCapabilities,
	build_openai_chat_payload,
	ensure_supported_request,
	extract_reasoning_mode,
	infer_capabilities_from_model_metadata,
)
from ..errors import ErrorType, GatewayError
from ..schemas import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse
from ..streaming.control import ProviderStreamControl


class GeminiProvider(BaseProvider):
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
			supports_audio_output=False,
			supports_reasoning=True,
			supports_response_format=True,
		)

	def _ensure_api_key(self) -> None:
		if not self.api_key:
			raise GatewayError(
				"Gemini API key is required",
				type=ErrorType.MISSING_API_KEY,
				provider_id=self.id,
				status_code=401,
			)

	def _model_input_modalities(self, model_id: str) -> List[str]:
		modalities = ["text"]
		if model_id.startswith("gemini-"):
			modalities.extend(["image", "audio"])
		return modalities

	def _model_output_modalities(self, model_id: str) -> List[str]:
		modalities = ["text"]
		if "image" in model_id:
			modalities.append("image")
		return modalities

	def _supported_parameters(self, model_data: Dict[str, Any]) -> List[str]:
		parameters = [
			"max_tokens",
			"temperature",
			"top_p",
			"stop",
			"presence_penalty",
			"frequency_penalty",
			"response_format",
			"seed",
			"tools",
			"tool_choice",
			"parallel_tool_calls",
		]

		if model_data.get("thinking"):
			parameters.extend(["reasoning", "reasoning_effort"])

		return parameters

	def _request_extras(self, request: ChatCompletionRequest) -> Dict[str, Any]:
		return dict(getattr(request, "model_extra", None) or {})

	def _google_options(self, request: ChatCompletionRequest) -> Dict[str, Any]:
		extras = self._request_extras(request)
		google_options: Dict[str, Any] = {}

		top_level_google = extras.get("google")
		if isinstance(top_level_google, dict):
			google_options.update(top_level_google)

		extra_body = extras.get("extra_body")
		if isinstance(extra_body, dict):
			nested = extra_body.get("extra_body")
			if isinstance(nested, dict):
				extra_body = nested
			google_from_body = extra_body.get("google")
			if isinstance(google_from_body, dict):
				google_options.update(google_from_body)

		return google_options

	def _should_use_native_api(self, request: ChatCompletionRequest) -> bool:
		google_options = self._google_options(request)
		return bool(google_options)

	def _native_model_name(self, model: str) -> str:
		if model.startswith("models/"):
			return model
		return f"models/{model}"

	def _parse_json_like(self, value: Any, *, fallback_key: str = "content") -> Any:
		if isinstance(value, (dict, list, int, float, bool)) or value is None:
			return value
		if isinstance(value, str):
			text = value.strip()
			if not text:
				return {}
			try:
				return json.loads(text)
			except json.JSONDecodeError:
				return {fallback_key: value}
		return {fallback_key: str(value)}

	def _data_url_to_inline_data(self, url: str) -> Dict[str, Any]:
		if not url.startswith("data:") or ";base64," not in url:
			raise GatewayError(
				"Gemini native API only accepts base64 data URLs for inline media parts",
				type=ErrorType.UNSUPPORTED_FEATURE,
				status_code=400,
			)

		prefix, data = url.split(",", 1)
		mime_type = prefix[5:].split(";", 1)[0] or "application/octet-stream"
		return {"mimeType": mime_type, "data": data}

	def _input_audio_to_inline_data(self, input_audio: Dict[str, Any]) -> Dict[str, Any]:
		data = input_audio.get("data")
		if not isinstance(data, str) or not data:
			raise GatewayError(
				"Gemini audio input requires base64-encoded audio data",
				type=ErrorType.UNSUPPORTED_FEATURE,
				status_code=400,
			)

		audio_format = (input_audio.get("format") or "wav").lower()
		mime_type = input_audio.get("mime_type") or f"audio/{audio_format}"
		return {"mimeType": mime_type, "data": data}

	def _message_parts_to_native(self, content: Any) -> List[Dict[str, Any]]:
		if isinstance(content, str):
			return [{"text": content}]

		parts: List[Dict[str, Any]] = []
		for part in content or []:
			part_type = part.get("type")
			if part_type == "text":
				parts.append({"text": part.get("text", "")})
				continue

			if part_type == "image_url":
				image_url = (part.get("image_url") or {}).get("url", "")
				parts.append({"inlineData": self._data_url_to_inline_data(image_url)})
				continue

			if part_type in {"input_audio", "audio"}:
				payload = part.get("input_audio") or part.get("audio") or {}
				parts.append({"inlineData": self._input_audio_to_inline_data(payload)})
				continue

			if part_type == "audio_url":
				audio_url = (part.get("audio_url") or {}).get("url", "")
				parts.append({"inlineData": self._data_url_to_inline_data(audio_url)})
				continue

			raise GatewayError(
				f"Gemini native API does not support content part type '{part_type}'",
				type=ErrorType.UNSUPPORTED_FEATURE,
				status_code=400,
			)

		return parts

	def _assistant_parts_to_native(self, request: ChatCompletionRequest, message: Any) -> List[Dict[str, Any]]:
		parts: List[Dict[str, Any]] = []
		if message.content is not None:
			parts.extend(self._message_parts_to_native(message.content))

		for index, tool_call in enumerate(message.tool_calls or []):
			function = tool_call.get("function") or {}
			parts.append(
				{
					"functionCall": {
						"id": tool_call.get("id") or f"call_{index}",
						"name": function.get("name", ""),
						"args": self._parse_json_like(function.get("arguments")),
					}
				}
			)

		return parts

	def _tool_message_to_native(self, message: Any) -> Dict[str, Any]:
		function_response = {
			"name": message.name or "tool",
			"response": self._parse_json_like(message.content),
		}
		if message.tool_call_id:
			function_response["id"] = message.tool_call_id

		return {"role": "user", "parts": [{"functionResponse": function_response}]}

	def _messages_to_native_contents(self, request: ChatCompletionRequest) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
		contents: List[Dict[str, Any]] = []
		system_texts: List[str] = []

		for message in request.messages:
			if message.role == "system":
				if isinstance(message.content, str) and message.content.strip():
					system_texts.append(message.content)
				continue

			if message.role == "assistant":
				parts = self._assistant_parts_to_native(request, message)
				if parts:
					contents.append({"role": "model", "parts": parts})
				continue

			if message.role == "tool":
				contents.append(self._tool_message_to_native(message))
				continue

			parts = self._message_parts_to_native(message.content)
			if message.audio:
				parts.append({"inlineData": self._input_audio_to_inline_data(message.audio)})
			if parts:
				contents.append({"role": "user", "parts": parts})

		system_instruction = None
		if system_texts:
			system_instruction = {"parts": [{"text": "\n\n".join(system_texts)}]}

		return contents, system_instruction

	def _native_tools(self, request: ChatCompletionRequest) -> Optional[List[Dict[str, Any]]]:
		function_declarations = []
		for tool in request.tools or []:
			if tool.get("type") != "function":
				continue
			function = dict(tool.get("function") or {})
			function_declarations.append(function)

		if not function_declarations:
			return None

		return [{"functionDeclarations": function_declarations}]

	def _native_tool_config(self, request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
		tool_choice = request.tool_choice
		if tool_choice is None:
			return None

		config: Dict[str, Any] = {}
		function_config: Dict[str, Any] = {}

		if tool_choice == "auto":
			function_config["mode"] = "AUTO"
		elif tool_choice == "none":
			function_config["mode"] = "NONE"
		elif tool_choice in {"required", "any"}:
			function_config["mode"] = "ANY"
		elif isinstance(tool_choice, dict):
			function_payload = tool_choice.get("function") or {}
			function_name = function_payload.get("name")
			function_config["mode"] = "ANY"
			if function_name:
				function_config["allowedFunctionNames"] = [function_name]
		else:
			function_config["mode"] = "AUTO"

		if function_config:
			config["functionCallingConfig"] = function_config

		return config or None

	def _native_generation_config(self, request: ChatCompletionRequest, google_options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		config = dict(google_options.get("generation_config") or {})

		if request.n is not None:
			config.setdefault("candidateCount", request.n)

		max_tokens = request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens
		if max_tokens is not None:
			config.setdefault("maxOutputTokens", max_tokens)

		if request.temperature is not None:
			config.setdefault("temperature", request.temperature)

		if request.top_p is not None:
			config.setdefault("topP", request.top_p)

		if request.presence_penalty is not None:
			config.setdefault("presencePenalty", request.presence_penalty)

		if request.frequency_penalty is not None:
			config.setdefault("frequencyPenalty", request.frequency_penalty)

		seed = self._request_extras(request).get("seed")
		if seed is not None:
			config.setdefault("seed", seed)

		if request.stop is not None:
			config.setdefault("stopSequences", request.stop if isinstance(request.stop, list) else [request.stop])

		if request.modalities:
			config.setdefault("responseModalities", [modality.upper() for modality in request.modalities])

		if request.response_format:
			format_type = request.response_format.get("type")
			if format_type == "json_object":
				config.setdefault("responseMimeType", "application/json")
			elif format_type == "json_schema":
				json_schema = request.response_format.get("json_schema") or {}
				schema = json_schema.get("schema") or json_schema
				config.setdefault("responseMimeType", "application/json")
				config.setdefault("responseJsonSchema", schema)

		google_thinking = dict(google_options.get("thinking_config") or {})
		reasoning_mode = extract_reasoning_mode(request)
		if reasoning_mode is not None and "thinkingLevel" not in google_thinking and "thinkingBudget" not in google_thinking:
			if reasoning_mode is False or reasoning_mode == "none":
				google_thinking["thinkingBudget"] = 0
			elif reasoning_mode is True:
				google_thinking["thinkingLevel"] = "HIGH"
			elif isinstance(reasoning_mode, str):
				google_thinking["thinkingLevel"] = reasoning_mode.upper()

		if google_thinking:
			config.setdefault("thinkingConfig", {})
			config["thinkingConfig"].update(google_thinking)

		return config or None

	def _native_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
		google_options = self._google_options(request)
		contents, system_instruction = self._messages_to_native_contents(request)

		if not contents:
			raise GatewayError(
				"Gemini native requests require at least one non-system content message",
				type=ErrorType.UNSUPPORTED_FEATURE,
				status_code=400,
			)

		payload: Dict[str, Any] = {"contents": contents}
		if system_instruction is not None:
			payload["systemInstruction"] = system_instruction

		tools = self._native_tools(request)
		if tools is not None:
			payload["tools"] = tools

		tool_config = self._native_tool_config(request)
		if tool_config is not None:
			payload["toolConfig"] = tool_config

		generation_config = self._native_generation_config(request, google_options)
		if generation_config is not None:
			payload["generationConfig"] = generation_config

		cached_content = google_options.get("cached_content")
		if cached_content:
			payload["cachedContent"] = cached_content

		safety_settings = google_options.get("safety_settings")
		if safety_settings is not None:
			payload["safetySettings"] = safety_settings

		service_tier = google_options.get("service_tier")
		if service_tier:
			payload["serviceTier"] = service_tier

		if google_options.get("store") is not None:
			payload["store"] = google_options["store"]

		return payload

	def _extract_native_message(self, provider_response: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str], Optional[Dict[str, Any]]]:
		candidate = (provider_response.get("candidates") or [{}])[0]
		content = candidate.get("content") or {}
		parts = content.get("parts") or []

		text_parts: List[str] = []
		reasoning_parts: List[str] = []
		tool_calls: List[Dict[str, Any]] = []
		audio: Optional[Dict[str, Any]] = None

		for index, part in enumerate(parts):
			text = part.get("text")
			if isinstance(text, str) and text:
				if part.get("thought"):
					reasoning_parts.append(text)
				else:
					text_parts.append(text)

			function_call = part.get("functionCall") or {}
			if function_call:
				tool_calls.append(
					{
						"id": function_call.get("id") or f"call_{index}",
						"type": "function",
						"function": {
							"name": function_call.get("name", ""),
							"arguments": json.dumps(function_call.get("args") or {}),
						},
					}
				)

			inline_data = part.get("inlineData") or {}
			mime_type = inline_data.get("mimeType", "")
			if mime_type.startswith("audio/") and inline_data.get("data"):
				audio = {
					"data": inline_data["data"],
					"format": mime_type.split("/", 1)[-1],
				}

		message_text = "".join(text_parts) if text_parts else None
		reasoning_text = "".join(reasoning_parts) if reasoning_parts else None
		return message_text, tool_calls or None, reasoning_text, audio

	def _native_finish_reason(self, provider_response: Dict[str, Any], tool_calls: Optional[List[Dict[str, Any]]]) -> str:
		candidate = (provider_response.get("candidates") or [{}])[0]
		finish_reason = (candidate.get("finishReason") or "").upper()

		if tool_calls:
			return "tool_calls"

		mapping = {
			"STOP": "stop",
			"MAX_TOKENS": "length",
			"SAFETY": "content_filter",
			"BLOCKLIST": "content_filter",
			"PROHIBITED_CONTENT": "content_filter",
			"SPII": "content_filter",
		}
		return mapping.get(finish_reason, "stop")

	def _native_usage(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
		usage = provider_response.get("usageMetadata") or {}
		return {
			"prompt_tokens": usage.get("promptTokenCount"),
			"completion_tokens": usage.get("candidatesTokenCount"),
			"total_tokens": usage.get("totalTokenCount"),
		}

	def _native_to_chat_response(self, provider_response: Dict[str, Any], model: str) -> ChatCompletionResponse:
		message_text, tool_calls, reasoning_text, audio = self._extract_native_message(provider_response)
		message: Dict[str, Any] = {"role": "assistant", "content": message_text}
		if tool_calls:
			message["tool_calls"] = tool_calls
		if reasoning_text:
			message["reasoning"] = reasoning_text
		if audio:
			message["audio"] = audio

		return ChatCompletionResponse(
			id=provider_response.get("responseId") or f"chatcmpl-{int(time.time() * 1000)}",
			created=int(time.time()),
			model=model,
			provider={
				"id": self.id,
				"type": self.type,
				"actual_model": provider_response.get("modelVersion") or model,
			},
			choices=[
				{
					"index": 0,
					"message": message,
					"finish_reason": self._native_finish_reason(provider_response, tool_calls),
				}
			],
			usage=self._native_usage(provider_response),
		)

	def _native_stream_chunk_to_openai(
		self,
		provider_chunk: Dict[str, Any],
		model: str,
		state: Dict[str, Any],
	) -> List[bytes]:
		message_text, tool_calls, reasoning_text, audio = self._extract_native_message(provider_chunk)
		response_id = provider_chunk.get("responseId") or state.get("id") or f"chatcmpl-{int(time.time() * 1000)}"
		state["id"] = response_id
		created = state.setdefault("created", int(time.time()))

		emitted: List[bytes] = []
		delta: Dict[str, Any] = {}

		previous_text = state.get("text", "")
		if message_text:
			if message_text.startswith(previous_text):
				text_delta = message_text[len(previous_text):]
			else:
				text_delta = message_text
			if text_delta:
				delta["content"] = text_delta
			state["text"] = message_text

		previous_reasoning = state.get("reasoning", "")
		if reasoning_text:
			if reasoning_text.startswith(previous_reasoning):
				reasoning_delta = reasoning_text[len(previous_reasoning):]
			else:
				reasoning_delta = reasoning_text
			if reasoning_delta:
				delta["reasoning"] = reasoning_delta
			state["reasoning"] = reasoning_text

		if tool_calls:
			known = state.setdefault("tool_calls", {})
			stream_tool_calls: List[Dict[str, Any]] = []
			for index, tool_call in enumerate(tool_calls):
				tool_id = tool_call.get("id") or f"call_{index}"
				arguments = ((tool_call.get("function") or {}).get("arguments")) or "{}"
				previous_arguments = known.get(tool_id, "")
				if arguments.startswith(previous_arguments):
					argument_delta = arguments[len(previous_arguments):]
				else:
					argument_delta = arguments
				if argument_delta or tool_id not in known:
					stream_tool_calls.append(
						{
							"index": index,
							"id": tool_id,
							"type": "function",
							"function": {
								"name": (tool_call.get("function") or {}).get("name", ""),
								"arguments": argument_delta,
							},
						}
					)
				known[tool_id] = arguments
			if stream_tool_calls:
				delta["tool_calls"] = stream_tool_calls

		if audio:
			previous_audio = state.get("audio") or {}
			audio_delta: Dict[str, Any] = {}
			for key, value in audio.items():
				previous_value = previous_audio.get(key)
				if isinstance(value, str) and isinstance(previous_value, str) and value.startswith(previous_value):
					change = value[len(previous_value):]
				else:
					change = value
				if change:
					audio_delta[key] = change
			if audio_delta:
				delta["audio"] = audio_delta
			state["audio"] = audio

		if delta:
			chunk = ChatCompletionChunk(
				id=response_id,
				created=created,
				model=model,
				choices=[{"index": 0, "delta": delta, "finish_reason": None}],
			)
			emitted.append(f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode())

		tool_calls_for_finish = tool_calls if tool_calls else None
		finish_reason = self._native_finish_reason(provider_chunk, tool_calls_for_finish)
		candidate = (provider_chunk.get("candidates") or [{}])[0]
		if candidate.get("finishReason"):
			chunk = ChatCompletionChunk(
				id=response_id,
				created=created,
				model=model,
				choices=[{"index": 0, "delta": {}, "finish_reason": finish_reason}],
			)
			emitted.append(f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode())

		usage = self._native_usage(provider_chunk)
		if any(value is not None for value in usage.values()):
			usage_chunk = {
				"id": response_id,
				"object": "chat.completion.chunk",
				"created": created,
				"model": model,
				"choices": [],
				"usage": usage,
			}
			emitted.append(f"data: {json.dumps(usage_chunk)}\n\n".encode())

		return emitted

	async def list_models(self) -> List[ProviderModel]:
		if time.time() - self._models_cache["timestamp"] < 300 and self._models_cache["models"]:
			return self._models_cache["models"]

		self._ensure_api_key()
		models: List[ProviderModel] = []
		page_token: Optional[str] = None

		async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
			while True:
				params: Dict[str, Any] = {"pageSize": 1000}
				if page_token:
					params["pageToken"] = page_token

				response = await client.get(
					f"{self.native_base_url}/models",
					headers=self.native_headers,
					params=params,
				)
				if response.status_code != 200:
					raise self.normalize_error(response)

				payload = response.json()
				for model_data in payload.get("models", []):
					supported_generation_methods = model_data.get("supportedGenerationMethods") or []
					if "generateContent" not in supported_generation_methods:
						continue

					model_id = model_data.get("baseModelId") or model_data.get("name", "").split("/", 1)[-1]
					input_modalities = self._model_input_modalities(model_id)
					output_modalities = self._model_output_modalities(model_id)
					supported_parameters = self._supported_parameters(model_data)

					models.append(
						ProviderModel(
							id=model_id,
							name=model_data.get("displayName", model_id),
							context_length=model_data.get("inputTokenLimit"),
							max_completion_tokens=model_data.get("outputTokenLimit"),
							input_modalities=input_modalities,
							output_modalities=output_modalities,
							supported_parameters=supported_parameters,
							capabilities=infer_capabilities_from_model_metadata(
								supported_parameters=supported_parameters,
								input_modalities=input_modalities,
								output_modalities=output_modalities,
							),
							raw=model_data,
						)
					)

				page_token = payload.get("nextPageToken")
				if not page_token:
					break

		self._models_cache = {"timestamp": time.time(), "models": models}
		return models

	async def health_check(self) -> ProviderHealth:
		start_time = time.time()
		try:
			await self.list_models()
			return ProviderHealth(healthy=True, latency_ms=(time.time() - start_time) * 1000)
		except Exception as error:
			return ProviderHealth(
				healthy=False,
				message=str(error),
				latency_ms=(time.time() - start_time) * 1000,
			)

	def convert_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
		self._ensure_api_key()
		ensure_supported_request(request, self.capabilities, self.id)
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
			response = await client.send(
				client.build_request(
					"POST",
					f"{self.base_url}/chat/completions",
					headers=self.headers,
					json=self.convert_request(stream_request),
				),
				stream=True,
			)

			if stream_control is not None:
				stream_control.register_cancel_callback(abort_stream, native_supported=True)

			if response.status_code != 200:
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
			response = await client.send(
				client.build_request(
					"POST",
					f"{self.native_base_url}/{self._native_model_name(request.model)}:streamGenerateContent?alt=sse",
					headers=self.native_headers,
					json=self._native_payload(request),
				),
				stream=True,
			)

			if stream_control is not None:
				stream_control.register_cancel_callback(abort_stream, native_supported=True)

			if response.status_code != 200:
				raise self.normalize_error(response)

			async for line in response.aiter_lines():
				if not line or not line.startswith("data:"):
					continue
				payload = line[5:].strip()
				if not payload:
					continue
				chunk_payload = json.loads(payload)
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