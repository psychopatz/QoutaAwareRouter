from abc import ABC, abstractmethod
from typing import List, AsyncIterator, Optional, Dict, Any
from pydantic import BaseModel, Field
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..errors import GatewayError
from ..streaming.control import ProviderStreamControl
from .openai_compatible import ProviderCapabilities, combine_capabilities, ensure_supported_request

class ProviderModel(BaseModel):
    id: str
    name: str
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    input_modalities: List[str] = Field(default_factory=list)
    output_modalities: List[str] = Field(default_factory=list)
    supported_parameters: List[str] = Field(default_factory=list)
    capabilities: Optional[ProviderCapabilities] = None
    raw: Dict[str, Any] = Field(default_factory=dict)

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
        self.capabilities = self.default_capabilities().model_copy(
            update=self.config.get("capabilities", {})
        )
        self.model_capability_overrides = self.config.get("model_capabilities", {})

    def default_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def _get_model_capability_override(self, model_id: str) -> Dict[str, Any]:
        return (
            self.model_capability_overrides.get(model_id)
            or self.model_capability_overrides.get("*")
            or {}
        )

    async def get_model_info(self, model_id: str) -> Optional[ProviderModel]:
        try:
            for model in await self.list_models():
                if model.id == model_id:
                    return model
        except Exception:
            return None
        return None

    def effective_capabilities_for_model(
        self,
        model_id: str,
        model_info: Optional[ProviderModel] = None,
    ) -> ProviderCapabilities:
        capabilities = self.capabilities.model_copy()

        if model_info and model_info.capabilities is not None:
            capabilities = combine_capabilities(capabilities, model_info.capabilities, mode="narrow")

        capability_override = self._get_model_capability_override(model_id)
        if capability_override:
            capabilities = capabilities.model_copy(update=capability_override)

        return capabilities

    async def validate_model_request(self, model_id: str, request: ChatCompletionRequest) -> ProviderCapabilities:
        model_info = await self.get_model_info(model_id)
        capabilities = self.effective_capabilities_for_model(model_id, model_info)
        ensure_supported_request(request, capabilities, f"{self.id}/{model_id}")
        return capabilities

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
    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        stream_control: Optional[ProviderStreamControl] = None,
    ) -> AsyncIterator[bytes]:
        """Send a streaming chat completion request."""
        pass

    async def test_key(self, api_key: str) -> bool:
        """Test if an API key is valid for this provider."""
        import traceback
        import logging
        logger = logging.getLogger("qarouter.providers")
        
        # Save old config key
        old_key = self.config.get("api_key")
        
        # Temporarily apply new key
        self.config["api_key"] = api_key
        try:
            # list_models is usually a quick authenticated GET request
            await self.list_models()
            return True
        except Exception as e:
            logger.warning(f"Key test failed for provider {self.id}: {e}")
            return False
        finally:
            # Restore config
            if old_key is not None:
                self.config["api_key"] = old_key
            else:
                self.config.pop("api_key", None)

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
