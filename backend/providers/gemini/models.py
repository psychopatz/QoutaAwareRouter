import time
from typing import Any, Dict, List, Optional

import httpx

from ..base import ProviderHealth, ProviderModel
from ..openai_compatible import ProviderCapabilities, infer_capabilities_from_model_metadata, request_uses_audio_output
from ...errors import ErrorType, GatewayError
from ...schemas import ChatCompletionRequest


class GeminiModelMixin:
	def _normalized_model_id(self, model_id: str) -> str:
		return (model_id or "").removeprefix("models/").lower()

	def _is_tts_model(self, model_id: str) -> bool:
		normalized = self._normalized_model_id(model_id)
		return "tts" in normalized

	def _input_modalities_for_model(self, model_id: str) -> List[str]:
		if self._is_tts_model(model_id):
			return ["text"]

		modalities = ["text"]
		if self._normalized_model_id(model_id).startswith("gemini-"):
			modalities.extend(["image", "audio"])
		return modalities

	def _output_modalities_for_model(self, model_id: str) -> List[str]:
		if self._is_tts_model(model_id):
			return ["audio"]

		modalities = ["text"]
		if "image" in self._normalized_model_id(model_id):
			modalities.append("image")
		return modalities

	def _capabilities_for_model_hint(self, model_id: str) -> ProviderCapabilities:
		return infer_capabilities_from_model_metadata(
			supported_parameters=self._supported_parameters({"thinking": not self._is_tts_model(model_id)}),
			input_modalities=self._input_modalities_for_model(model_id),
			output_modalities=self._output_modalities_for_model(model_id),
		)

	def _ensure_api_key(self) -> None:
		if not self.api_key:
			raise GatewayError(
				"Gemini API key is required",
				type=ErrorType.MISSING_API_KEY,
				provider_id=self.id,
				status_code=401,
			)

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
		return bool(google_options) or request_uses_audio_output(request) or self._is_tts_model(request.model)

	def _native_model_name(self, model: str) -> str:
		if model.startswith("models/"):
			return model
		return f"models/{model}"

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
					input_modalities = self._input_modalities_for_model(model_id)
					output_modalities = self._output_modalities_for_model(model_id)
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