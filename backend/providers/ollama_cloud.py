import httpx
import json
import time
from typing import List, AsyncIterator, Dict, Any, Optional
from .base import BaseProvider, ProviderModel, ProviderHealth
from ..schemas import ChatCompletionRequest, ChatCompletionResponse, ResponseMessage, Choice, Usage, ProviderMetadata
from ..errors import GatewayError, ErrorType
from ..logging_config import logger
from ..streaming.control import ProviderStreamControl
from .openai_compatible import (
    ProviderCapabilities,
    convert_messages_to_ollama,
    ensure_supported_request,
    extract_reasoning_mode,
    infer_capabilities_from_model_metadata,
    normalize_tool_calls_from_ollama,
    ollama_response_format,
)

class OllamaCloudProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.config.get("base_url", "https://ollama.com").rstrip("/")
        self._models_cache = {"timestamp": 0.0, "models": []}

    @property
    def api_key(self) -> Optional[str]:
        return self.config.get("api_key") or self.config.get("api_key_env")

    def default_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_tool_choice=False,
            supports_parallel_tool_calls=False,
            supports_vision_input=True,
            supports_audio_input=False,
            supports_audio_output=False,
            supports_reasoning=True,
            supports_response_format=True,
        )

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # If it's an env var name, get it. Otherwise use as literal.
            import os
            key = os.getenv(self.api_key) if self.api_key in os.environ else self.api_key
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def _cached_model_info(self, model_id: str):
        for model in self._models_cache.get("models", []):
            if model.id == model_id:
                return model
        return None

    def _capabilities_for_model_hint(self, model_id: str) -> ProviderCapabilities:
        return self.effective_capabilities_for_model(model_id, self._cached_model_info(model_id))

    def _normalize_show_capabilities(self, capabilities: Any) -> set[str]:
        normalized = set()
        for capability in capabilities or []:
            if not isinstance(capability, str):
                continue
            normalized.add(capability.strip().lower().replace("-", "_").replace(" ", "_"))
        return normalized

    def _extract_context_length(self, show_data: Dict[str, Any]) -> Optional[int]:
        model_info = show_data.get("model_info") or {}
        for key, value in model_info.items():
            if not isinstance(key, str) or not key.endswith(".context_length"):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _infer_model_capabilities(self, show_data: Dict[str, Any]) -> ProviderCapabilities:
        capability_names = self._normalize_show_capabilities(show_data.get("capabilities"))
        if not capability_names:
            return self.capabilities.model_copy()

        supported_parameters: List[str] = []
        input_modalities = ["text"]

        if {"tools", "tool_calling", "tool"} & capability_names:
            supported_parameters.append("tools")
            supported_parameters.append("parallel_tool_calls")

        if {"thinking", "reasoning"} & capability_names:
            supported_parameters.append("reasoning_effort")

        if {"vision", "image", "multimodal"} & capability_names:
            input_modalities.append("image")

        inferred = infer_capabilities_from_model_metadata(
            supported_parameters=supported_parameters,
            input_modalities=input_modalities,
            output_modalities=["text"],
        )

        return self.capabilities.model_copy(
            update={
                "supports_tools": inferred.supports_tools,
                "supports_parallel_tool_calls": inferred.supports_parallel_tool_calls,
                "supports_vision_input": inferred.supports_vision_input,
                "supports_reasoning": inferred.supports_reasoning if inferred.supports_reasoning else self.capabilities.supports_reasoning,
            }
        )

    def _supported_parameters_for_capabilities(self, capabilities: ProviderCapabilities) -> List[str]:
        supported_parameters: List[str] = []
        if capabilities.supports_tools:
            supported_parameters.append("tools")
        if capabilities.supports_parallel_tool_calls:
            supported_parameters.append("parallel_tool_calls")
        if capabilities.supports_reasoning:
            supported_parameters.append("reasoning_effort")
        if capabilities.supports_response_format:
            supported_parameters.append("response_format")
        return supported_parameters

    async def _fetch_show_data(self, client: httpx.AsyncClient, model_id: str) -> Dict[str, Any]:
        try:
            response = await client.post(
                f"{self.base_url}/api/show",
                headers=self._get_headers(),
                json={"model": model_id},
            )
            response.raise_for_status()
            return response.json()
        except Exception as error:
            logger.debug(f"Failed to fetch Ollama show metadata for {self.id}/{model_id}: {error}")
            return {}

    async def list_models(self) -> List[ProviderModel]:
        if time.time() - self._models_cache["timestamp"] < 300 and self._models_cache["models"]:
            return self._models_cache["models"]

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/api/tags", headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                listed_models = data.get("models", [])

                models = []
                for model_data in listed_models:
                    details = model_data.get("details") or {}
                    show_data = await self._fetch_show_data(client, model_data["name"])
                    capabilities = self._infer_model_capabilities(show_data)
                    models.append(
                        ProviderModel(
                            id=model_data["name"],
                            name=model_data["name"],
                            context_length=self._extract_context_length(show_data),
                            input_modalities=["text", "image"] if capabilities.supports_vision_input else ["text"],
                            output_modalities=["text"],
                            supported_parameters=self._supported_parameters_for_capabilities(capabilities),
                            capabilities=capabilities,
                            raw={"details": details, "digest": model_data.get("digest"), "show": show_data},
                        )
                    )

                self._models_cache = {"timestamp": time.time(), "models": models}
                return models
        except Exception as e:
            logger.error(f"Failed to list models for provider {self.id}: {e}")
            return []

    async def health_check(self) -> ProviderHealth:
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags", headers=self._get_headers())
                latency = (time.time() - start_time) * 1000
                if response.status_code == 200:
                    return ProviderHealth(healthy=True, latency_ms=latency)
                return ProviderHealth(healthy=False, message=f"HTTP {response.status_code}", latency_ms=latency)
        except Exception as e:
            return ProviderHealth(healthy=False, message=str(e))

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = self.convert_request(request)
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=self._get_headers()
                )
                if response.status_code != 200:
                    raise self.normalize_error(response)
                
                return self.convert_response(response.json(), request.model)
        except httpx.HTTPError as e:
            raise self.normalize_error(e)

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        stream_control: Optional[ProviderStreamControl] = None,
    ) -> AsyncIterator[bytes]:
        payload = self.convert_request(request)
        payload["stream"] = True

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
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=self._get_headers(),
                ),
                stream=True,
            )

            if stream_control is not None:
                stream_control.register_cancel_callback(abort_stream, native_supported=False)

            if response.status_code != 200:
                error_text = await response.aread()
                raise GatewayError(f"Ollama Cloud error: {error_text.decode()}", status_code=response.status_code)

            async for line in response.aiter_lines():
                if line:
                    yield self.convert_stream_chunk(line.encode())
        finally:
            if response is not None:
                await response.aclose()
            await client.aclose()

    def convert_request(self, openai_request: ChatCompletionRequest) -> Dict[str, Any]:
        ensure_supported_request(
            openai_request,
            self._capabilities_for_model_hint(openai_request.model),
            f"{self.id}/{openai_request.model}",
        )
        messages = convert_messages_to_ollama(openai_request.messages)

        payload = {
            "model": openai_request.model,
            "messages": messages,
            "stream": False,  # Default to false, overridden in stream_chat_completion
            "options": {
                "temperature": openai_request.temperature,
                "top_p": openai_request.top_p,
                "stop": openai_request.stop,
                "num_predict": openai_request.max_completion_tokens or openai_request.max_tokens,
            }
        }
        
        if openai_request.tools and openai_request.tool_choice != "none":
            payload["tools"] = openai_request.tools

        response_format = ollama_response_format(openai_request)
        if response_format is not None:
            payload["format"] = response_format

        reasoning_mode = extract_reasoning_mode(openai_request)
        if reasoning_mode is not None:
            payload["think"] = reasoning_mode
            
        return payload

    def convert_response(self, provider_response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        # Ollama /api/chat response:
        # { "model": "...", "message": {"role": "assistant", "content": "..."}, "done": True, ... }
        message_data = provider_response.get("message", {})
        
        tool_calls = normalize_tool_calls_from_ollama(message_data.get("tool_calls"))
        reasoning = message_data.get("thinking")
                    
        return ChatCompletionResponse(
            id=f"chatcmpl-qarouter-{int(time.time())}",
            created=int(time.time()),
            model=model,
            provider=ProviderMetadata(
                id=self.id,
                type=self.type,
                actual_model=provider_response.get("model", "unknown")
            ),
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content"),
                        tool_calls=tool_calls,
                        reasoning=reasoning,
                    ),
                    finish_reason="tool_calls" if tool_calls else (provider_response.get("done_reason") or ("stop" if provider_response.get("done") else None))
                )
            ],
            usage=Usage(
                prompt_tokens=provider_response.get("prompt_eval_count"),
                completion_tokens=provider_response.get("eval_count"),
                total_tokens=(provider_response.get("prompt_eval_count") or 0) + (provider_response.get("eval_count") or 0)
            )
        )

    def convert_stream_chunk(self, provider_chunk: bytes) -> bytes:
        # Ollama NDJSON: {"model":"...","message":{"role":"assistant","content":"..."},"done":false}
        try:
            data = json.loads(provider_chunk)
            message_data = data.get("message", {})
            content = message_data.get("content", "")
            
            tool_calls = normalize_tool_calls_from_ollama(message_data.get("tool_calls"))
            
            delta = {"content": content}
            if tool_calls:
                delta["tool_calls"] = tool_calls
            if message_data.get("thinking"):
                delta["reasoning"] = message_data.get("thinking")
                
            chunk = {
                "id": f"chatcmpl-qarouter-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "tool_calls" if tool_calls else ("stop" if data.get("done") else None)
                }]
            }
            res = f"data: {json.dumps(chunk)}\n\n"
            
            if data.get("done"):
                # Append usage chunk immediately after
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                usage_chunk = {
                    "id": chunk["id"],
                    "object": "chat.completion.chunk",
                    "created": chunk["created"],
                    "model": chunk["model"],
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                res += f"data: {json.dumps(usage_chunk)}\n\n"
                res += "data: [DONE]\n\n"
                
            return res.encode()
        except Exception:
            return b""

    def normalize_error(self, error: Any) -> GatewayError:
        if isinstance(error, httpx.Response):
            status_code = error.status_code
            try:
                body = error.json()
                message = body.get("error", str(error.text))
            except:
                message = str(error.text)
            
            if self._is_quota_error(status_code, message):
                return GatewayError(message, type=ErrorType.RATE_LIMITED, provider_id=self.id, status_code=status_code)
            
            if status_code == 401 or status_code == 403:
                return GatewayError(message, type=ErrorType.AUTH_FAILED, provider_id=self.id, status_code=status_code)
            
            return GatewayError(message, status_code=status_code)
        
        return GatewayError(str(error))
