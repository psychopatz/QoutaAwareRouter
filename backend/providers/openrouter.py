import time
from typing import List, AsyncIterator, Dict, Any, Optional
import httpx

from .base import BaseProvider, ProviderModel, ProviderHealth
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..errors import GatewayError, ErrorType
from ..streaming.control import ProviderStreamControl
from .openai_compatible import (
    ProviderCapabilities,
    build_openai_chat_payload,
    ensure_supported_request,
    infer_capabilities_from_model_metadata,
)

class OpenRouterProvider(BaseProvider):
    def __init__(self, **kwargs):
        kwargs.setdefault('supports_streaming', True)
        super().__init__(**kwargs)
        self.base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        self._models_cache = {"timestamp": 0.0, "models": []}

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
        
    @property
    def api_key(self) -> str:
        return self.config.get("api_key", "")
        
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:7317",
            "X-Title": "Quota Aware LLM Router",
            "Content-Type": "application/json"
        }
        
    async def list_models(self) -> List[ProviderModel]:
        if time.time() - self._models_cache["timestamp"] < 300 and self._models_cache["models"]:
            return self._models_cache["models"]

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/models", headers=self.headers, timeout=10.0)
            if resp.status_code != 200:
                raise self.normalize_error(ValueError(f"Status {resp.status_code}: {resp.text}"))
            data = resp.json()
            models = []
            for model_data in data.get("data", []):
                architecture = model_data.get("architecture") or {}
                top_provider = model_data.get("top_provider") or {}
                supported_parameters = model_data.get("supported_parameters") or []
                input_modalities = architecture.get("input_modalities") or []
                output_modalities = architecture.get("output_modalities") or []

                models.append(
                    ProviderModel(
                        id=model_data["id"],
                        name=model_data.get("name", model_data["id"]),
                        context_length=model_data.get("context_length") or top_provider.get("context_length"),
                        max_completion_tokens=top_provider.get("max_completion_tokens"),
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

            self._models_cache = {"timestamp": time.time(), "models": models}
            return models
            
    async def health_check(self) -> ProviderHealth:
        try:
            # OpenRouter occasionally struggles, pinging models endpoint proves connectivity
            async with httpx.AsyncClient() as client:
                res = await client.get("https://openrouter.ai/api/v1/models", timeout=5.0)
                if res.status_code == 200:
                    return ProviderHealth(healthy=True)
                return ProviderHealth(healthy=False, message=res.text)
        except Exception as e:
            return ProviderHealth(healthy=False, message=str(e))
            
    def convert_request(self, req: ChatCompletionRequest) -> Dict[str, Any]:
        ensure_supported_request(req, self.capabilities, self.id)
        return build_openai_chat_payload(req)
        
    def convert_response(self, resp: Dict[str, Any], model: str) -> ChatCompletionResponse:
        response_payload = dict(resp)
        response_payload["model"] = model
        response_payload["provider"] = {
            "id": self.id,
            "type": self.type,
            "actual_model": resp.get("model", model),
        }
        return ChatCompletionResponse(**response_payload)
        
    def convert_stream_chunk(self, chunk: bytes) -> bytes:
        # Simply proxy SSE stream chunks
        return chunk
        
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=self.convert_request(request),
                timeout=self.timeout_seconds
            )
            if resp.status_code != 200:
                raise self.normalize_error(ValueError(f"Status {resp.status_code}: {resp.text}"))
            return self.convert_response(resp.json(), request.model)
            
    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        stream_control: Optional[ProviderStreamControl] = None,
    ) -> AsyncIterator[bytes]:
        request = request.model_copy(update={"stream": True})
        client = httpx.AsyncClient()
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
                    json=self.convert_request(request),
                ),
                stream=True,
                timeout=self.timeout_seconds,
            )

            if stream_control is not None:
                stream_control.register_cancel_callback(abort_stream, native_supported=True)

            if response.status_code != 200:
                error_body = await response.aread()
                raise self.normalize_error(ValueError(f"Stream error {response.status_code}: {error_body.decode(errors='ignore')}"))

            async for chunk in response.aiter_bytes():
                yield self.convert_stream_chunk(chunk)
        finally:
            if response is not None:
                await response.aclose()
            await client.aclose()
                    
    def normalize_error(self, error: Any) -> GatewayError:
        error_msg = str(error)
        status_code = getattr(error, 'status_code', 500)
        
        if "Status 429" in error_msg or self._is_quota_error(status_code, error_msg):
            return GatewayError(
                error_msg,
                type=ErrorType.RATE_LIMITED,
                provider_id=self.id,
                status_code=status_code,
            )
            
        return GatewayError(error_msg, provider_id=self.id, status_code=status_code)
