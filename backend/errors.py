from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel

class ErrorType(str, Enum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    ALL_PROVIDERS_UNAVAILABLE = "all_providers_unavailable"
    QUOTA_LIMITED = "quota_limited"
    RATE_LIMITED = "rate_limited"
    UNSUPPORTED_FEATURE = "unsupported_feature"
    AUTH_FAILED = "auth_failed"
    MISSING_API_KEY = "missing_api_key"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_MODEL = "invalid_model"
    TIMEOUT = "timeout"
    GATEWAY_CONFIG_ERROR = "gateway_config_error"
    PROVIDER_RESPONSE_ERROR = "provider_response_error"

class ErrorDetail(BaseModel):
    type: ErrorType
    message: str
    provider_id: Optional[str] = None
    retry_after_seconds: Optional[int] = None

class GatewayError(Exception):
    def __init__(
        self, 
        message: str, 
        type: ErrorType = ErrorType.PROVIDER_RESPONSE_ERROR,
        provider_id: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        status_code: int = 500
    ):
        self.message = message
        self.type = type
        self.provider_id = provider_id
        self.retry_after_seconds = retry_after_seconds
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> dict:
        error = {
            "type": self.type.value,
            "message": self.message,
        }
        if self.provider_id:
            error["provider_id"] = self.provider_id
        if self.retry_after_seconds:
            error["retry_after_seconds"] = self.retry_after_seconds
        return {"error": error}
