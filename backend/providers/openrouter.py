from typing import List, AsyncIterator, Dict, Any
import httpx

from .base import BaseProvider, ProviderModel, ProviderHealth
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..errors import GatewayError

class OpenRouterProvider(BaseProvider):
    def __init__(self, **kwargs):
        kwargs.setdefault('supports_streaming', True)
        super().__init__(**kwargs)
        self.base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")
        
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
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/models", headers=self.headers, timeout=10.0)
            if resp.status_code != 200:
                raise self.normalize_error(ValueError(f"Status {resp.status_code}: {resp.text}"))
            data = resp.json()
            return [ProviderModel(id=m["id"], name=m.get("name", m["id"])) for m in data.get("data", [])]
            
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
        return req.model_dump(exclude_unset=True)
        
    def convert_response(self, resp: Dict[str, Any], model: str) -> ChatCompletionResponse:
        return ChatCompletionResponse(**resp)
        
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
            
    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        request.stream = True
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", 
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=self.convert_request(request),
                timeout=self.timeout_seconds
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise self.normalize_error(ValueError(f"Stream error {response.status_code}: {response.text}"))
                    
                async for chunk in response.aiter_bytes():
                    yield self.convert_stream_chunk(chunk)
                    
    def normalize_error(self, error: Any) -> GatewayError:
        error_msg = str(error)
        status_code = getattr(error, 'status_code', 500)
        
        if "Status 429" in error_msg or self._is_quota_error(status_code, error_msg):
            from ..errors import GatewayRateLimitError
            return GatewayRateLimitError(error_msg, "openrouter")
            
        return GatewayError(error_msg, status_code=status_code)
