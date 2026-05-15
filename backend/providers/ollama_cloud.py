import httpx
import json
import time
from typing import List, AsyncIterator, Dict, Any, Optional
from .base import BaseProvider, ProviderModel, ProviderHealth
from ..schemas import ChatCompletionRequest, ChatCompletionResponse, ResponseMessage, Choice, Usage, ProviderMetadata
from ..errors import GatewayError, ErrorType
from ..logging_config import logger

class OllamaCloudProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.config.get("base_url", "https://ollama.com").rstrip("/")
        self.api_key = self.config.get("api_key") or self.config.get("api_key_env")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # If it's an env var name, get it. Otherwise use as literal.
            import os
            key = os.getenv(self.api_key) if self.api_key in os.environ else self.api_key
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    async def list_models(self) -> List[ProviderModel]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/api/tags", headers=self._get_headers())
                response.raise_for_status()
                data = response.json()
                return [ProviderModel(id=m["name"], name=m["name"]) for m in data.get("models", [])]
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

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        payload = self.convert_request(request)
        payload["stream"] = True
        
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async with client.stream(
                "POST", 
                f"{self.base_url}/api/chat", 
                json=payload, 
                headers=self._get_headers()
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise GatewayError(f"Ollama Cloud error: {error_text.decode()}", status_code=response.status_code)
                
                async for line in response.aiter_lines():
                    if line:
                        yield self.convert_stream_chunk(line.encode())

    def convert_request(self, openai_request: ChatCompletionRequest) -> Dict[str, Any]:
        # Resolve model: the router usually passes the actual provider model here
        return {
            "model": openai_request.model,
            "messages": [m.model_dump() for m in openai_request.messages],
            "stream": False,  # Default to false, overridden in stream_chat_completion
            "options": {
                "temperature": openai_request.temperature,
                "top_p": openai_request.top_p,
                "stop": openai_request.stop,
                "num_predict": openai_request.max_tokens,
            }
        }

    def convert_response(self, provider_response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        # Ollama /api/chat response:
        # { "model": "...", "message": {"role": "assistant", "content": "..."}, "done": True, ... }
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
                        role=provider_response["message"]["role"],
                        content=provider_response["message"]["content"]
                    ),
                    finish_reason="stop" if provider_response.get("done") else None
                )
            ],
            usage=Usage(
                prompt_tokens=provider_response.get("prompt_eval_count"),
                completion_tokens=provider_response.get("eval_count"),
                total_tokens=None
            )
        )

    def convert_stream_chunk(self, provider_chunk: bytes) -> bytes:
        # Ollama NDJSON: {"model":"...","message":{"role":"assistant","content":"..."},"done":false}
        try:
            data = json.loads(provider_chunk)
            content = data.get("message", {}).get("content", "")
            
            chunk = {
                "id": f"chatcmpl-qarouter-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": data.get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": "stop" if data.get("done") else None
                }]
            }
            return f"data: {json.dumps(chunk)}\n\n".encode()
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
