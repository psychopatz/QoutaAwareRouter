from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional, Dict, Any
from pydantic import BaseModel
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..errors import GatewayError

class ProviderModel(BaseModel):
    id: str
    name: str

class ProviderHealth(BaseModel):
    healthy: bool
    message: Optional[str] = None
    latency_ms: Optional[float] = None

class BaseProvider(ABC):
    def __init__(
        self, 
        id: str, 
        type: str, 
        enabled: bool = True, 
        priority: int = 10, 
        supported_models: List[str] = None,
        supports_streaming: bool = True,
        max_concurrent_requests: int = 1,
        timeout_seconds: int = 120,
        cooldown_on_429_seconds: int = 300,
        **kwargs
    ):
        self.id = id
        self.type = type
        self.enabled = enabled
        self.priority = priority
        self.supported_models = supported_models or []
        self.supports_streaming = supports_streaming
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout_seconds = timeout_seconds
        self.cooldown_on_429_seconds = cooldown_on_429_seconds
        self.config = kwargs

    @abstractmethod
    async def list_models(self) -> List[ProviderModel]:
        """List models available on this provider."""
        pass

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check provider health."""
        pass

    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request."""
        pass

    @abstractmethod
    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Send a streaming chat completion request."""
        pass

    @abstractmethod
    def convert_request(self, openai_request: ChatCompletionRequest) -> Dict[str, Any]:
        """Convert OpenAI request to provider-specific format."""
        pass

    @abstractmethod
    def convert_response(self, provider_response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Convert provider-specific response to OpenAI format."""
        pass

    @abstractmethod
    def convert_stream_chunk(self, provider_chunk: bytes) -> bytes:
        """Convert provider-specific stream chunk to OpenAI SSE format."""
        pass

    def _is_quota_error(self, status_code: int, text: str) -> bool:
        text = text.lower()
        return (
            status_code == 429
            or "too many requests" in text
            or "rate limit" in text
            or "weekly usage limit" in text
            or "usage limit" in text
            or "quota" in text
            or "upgrade for higher limits" in text
            or ("reached your" in text and "limit" in text)
        )

    @abstractmethod
    def normalize_error(self, error: Any) -> GatewayError:
        """Normalize provider-specific errors to GatewayError."""
        pass
