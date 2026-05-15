from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import openai_compat
from backend.errors import ErrorType, GatewayError
from backend.providers.base import BaseProvider, ProviderHealth, ProviderModel
from backend.providers.registry import registry
from backend.storage import sqlite as sqlite_module
from backend.storage.sqlite import Storage


class DynamicListingProvider(BaseProvider):
    def __init__(self, listed_models: List[ProviderModel], required_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._listed_models = listed_models
        self.required_api_key = required_api_key

    async def list_models(self) -> List[ProviderModel]:
        if self.required_api_key and self.config.get("api_key") != self.required_api_key:
            raise GatewayError("missing api key", type=ErrorType.AUTH_FAILED, status_code=401)
        return self._listed_models

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True)

    async def chat_completion(self, request):
        raise NotImplementedError()

    async def stream_chat_completion(self, request, stream_control=None) -> AsyncIterator[bytes]:
        if False:
            yield b""

    def convert_request(self, openai_request) -> Dict[str, Any]:
        raise NotImplementedError()

    def convert_response(self, provider_response: Dict[str, Any], model: str):
        raise NotImplementedError()

    def convert_stream_chunk(self, provider_chunk: bytes) -> bytes:
        raise NotImplementedError()

    def normalize_error(self, error: Any) -> GatewayError:
        if isinstance(error, GatewayError):
            return error
        return GatewayError(str(error), type=ErrorType.PROVIDER_RESPONSE_ERROR)


def test_models_endpoint_uses_dynamic_keys_and_invalidation(tmp_path, monkeypatch):
    storage = Storage(db_path=str(tmp_path / "test.db"), key_file=str(tmp_path / "test.key"))
    monkeypatch.setattr(sqlite_module, "storage", storage)

    original_instances = dict(registry._active_instances)
    try:
        registry._active_instances = {
            "ollama-primary": DynamicListingProvider(
                id="ollama-primary",
                type="ollama_cloud",
                listed_models=[ProviderModel(id="llama3.1", name="llama3.1")],
            ),
            "openrouter-primary": DynamicListingProvider(
                id="openrouter-primary",
                type="openrouter",
                required_api_key="sk-openrouter",
                listed_models=[
                    ProviderModel(
                        id="openrouter/free",
                        name="Free Router",
                        raw={
                            "description": "Routes to free OpenRouter models.",
                            "pricing": {"prompt": "0", "completion": "0"},
                        },
                    )
                ],
            ),
        }
        openai_compat.invalidate_models_cache()

        app = FastAPI()
        app.include_router(openai_compat.router, prefix="/v1")
        client = TestClient(app)

        initial_response = client.get("/v1/models")
        assert initial_response.status_code == 200
        initial_models = initial_response.json()["data"]
        assert {model["owned_by"] for model in initial_models} == {"ollama_cloud"}

        storage.add_key("openrouter", "sk-openrouter")

        cached_response = client.get("/v1/models")
        cached_models = cached_response.json()["data"]
        assert {model["owned_by"] for model in cached_models} == {"ollama_cloud"}

        openai_compat.invalidate_models_cache()
        refreshed_response = client.get("/v1/models/openrouter")
        assert refreshed_response.status_code == 200
        refreshed_models = refreshed_response.json()["data"]
        assert len(refreshed_models) == 1
        assert refreshed_models[0]["owned_by"] == "openrouter"
        assert refreshed_models[0]["pricing"] == {"prompt": "0", "completion": "0"}
        assert refreshed_models[0]["is_free"] is True
        assert refreshed_models[0]["description"] == "Routes to free OpenRouter models."
    finally:
        registry._active_instances = original_instances
        openai_compat.invalidate_models_cache()