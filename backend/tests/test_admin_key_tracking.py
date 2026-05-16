from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api import admin
from backend.errors import ErrorType, GatewayError
from backend.providers.base import BaseProvider, ProviderHealth, ProviderModel
from backend.providers.openai_compatible import ProviderCapabilities
from backend.providers.registry import registry
from backend.routing.model_aliases import ModelAliasManager
from backend.routing.router import Router
from backend.schemas import ChatCompletionRequest, ChatCompletionResponse
from backend.storage import sqlite as sqlite_module
from backend.storage.sqlite import Storage


class QuotaFailProvider(BaseProvider):
    def default_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_tool_choice=True,
            supports_parallel_tool_calls=True,
            supports_vision_input=True,
            supports_audio_input=False,
            supports_audio_output=False,
            supports_reasoning=True,
            supports_response_format=True,
        )

    async def list_models(self):
        return [ProviderModel(id=self.supported_models[0], name=self.supported_models[0])]

    async def health_check(self):
        return ProviderHealth(healthy=True)

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise GatewayError(
            "Quota exhausted for this key. Upgrade for higher limits.",
            type=ErrorType.RATE_LIMITED,
            provider_id=self.id,
            status_code=429,
        )

    async def stream_chat_completion(self, request: ChatCompletionRequest, stream_control=None):
        yield b""

    def convert_request(self, openai_request: ChatCompletionRequest):
        return {}

    def convert_response(self, provider_response, model: str) -> ChatCompletionResponse:
        raise NotImplementedError

    def convert_stream_chunk(self, provider_chunk: bytes):
        return provider_chunk

    def normalize_error(self, error):
        if isinstance(error, GatewayError):
            return error
        return GatewayError(str(error))


class SuccessProvider(BaseProvider):
    def default_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def list_models(self):
        return [ProviderModel(id=self.supported_models[0], name=self.supported_models[0])]

    async def health_check(self):
        return ProviderHealth(healthy=True)

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    async def stream_chat_completion(self, request: ChatCompletionRequest, stream_control=None):
        yield b""

    async def test_key(self, api_key: str) -> bool:
        return api_key.endswith("-valid-key")

    def convert_request(self, openai_request: ChatCompletionRequest):
        return {}

    def convert_response(self, provider_response, model: str) -> ChatCompletionResponse:
        raise NotImplementedError

    def convert_stream_chunk(self, provider_chunk: bytes):
        return provider_chunk

    def normalize_error(self, error):
        if isinstance(error, GatewayError):
            return error
        return GatewayError(str(error))


class DynamicThenStaticProvider(BaseProvider):
    def default_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def list_models(self):
        return [ProviderModel(id=self.supported_models[0], name=self.supported_models[0])]

    async def health_check(self):
        return ProviderHealth(healthy=True)

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        current_key = self.config.get("api_key")
        if current_key == "dynamic-key":
            raise GatewayError(
                "Quota exhausted for this key. Upgrade for higher limits.",
                type=ErrorType.RATE_LIMITED,
                provider_id=self.id,
                status_code=429,
            )

        return ChatCompletionResponse(
            id="chatcmpl-static-success",
            created=123,
            model=request.model,
            provider={"id": self.id, "type": self.type, "actual_model": request.model},
            choices=[{"index": 0, "message": {"role": "assistant", "content": "static fallback success"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def stream_chat_completion(self, request: ChatCompletionRequest, stream_control=None):
        yield b""

    def convert_request(self, openai_request: ChatCompletionRequest):
        return {}

    def convert_response(self, provider_response, model: str) -> ChatCompletionResponse:
        raise NotImplementedError

    def convert_stream_chunk(self, provider_chunk: bytes):
        return provider_chunk

    def normalize_error(self, error):
        if isinstance(error, GatewayError):
            return error
        return GatewayError(str(error))


@pytest.fixture
def registry_backup():
    previous = dict(registry.get_all_instances())
    yield previous
    registry._active_instances = previous


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    temp_storage = Storage(
        db_path=str(tmp_path / "qarouter-test.db"),
        key_file=str(tmp_path / "qarouter-test.key"),
    )
    monkeypatch.setattr(sqlite_module, "storage", temp_storage)
    monkeypatch.setattr(admin, "storage", temp_storage)
    return temp_storage


@pytest.mark.asyncio
async def test_router_tracks_key_usage_and_quota_exhaustion(registry_backup, isolated_storage):
    isolated_storage.add_key("openrouter", "sk-test-1234")

    registry._active_instances = {
        "openrouter-primary": QuotaFailProvider(
            id="openrouter-primary",
            type="openrouter",
            supported_models=["openai/gpt-4o-mini"],
        )
    }

    alias_manager = ModelAliasManager(
        {
            "default": {
                "candidates": [
                    {"provider": "openrouter-primary", "model": "openai/gpt-4o-mini"},
                ]
            }
        }
    )
    router = Router(alias_manager, {})

    with pytest.raises(GatewayError):
        await router.route(
            ChatCompletionRequest(
                model="default",
                messages=[{"role": "user", "content": "hello"}],
            )
        )

    stored_key = isolated_storage.get_all_keys()[0]
    assert stored_key.request_count == 1
    assert stored_key.last_used_provider_id == "openrouter-primary"
    assert stored_key.last_used_model == "openai/gpt-4o-mini"
    assert stored_key.status == "quota_exhausted"
    assert stored_key.last_status_message == "Quota exhausted for this key. Upgrade for higher limits."
    assert stored_key.exhausted_at is not None


@pytest.mark.asyncio
async def test_router_falls_back_to_static_key_after_dynamic_key_exhaustion(registry_backup, isolated_storage):
    isolated_storage.add_key("openrouter", "dynamic-key")

    registry._active_instances = {
        "openrouter-primary": DynamicThenStaticProvider(
            id="openrouter-primary",
            type="openrouter",
            supported_models=["openai/gpt-4o-mini"],
            api_key="static-key",
        )
    }

    alias_manager = ModelAliasManager(
        {
            "default": {
                "candidates": [
                    {"provider": "openrouter-primary", "model": "openai/gpt-4o-mini"},
                ]
            }
        }
    )
    router = Router(alias_manager, {})

    response = await router.route(
        ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.content == "static fallback success"
    stored_key = isolated_storage.get_all_keys()[0]
    assert stored_key.status == "quota_exhausted"


def test_admin_key_summary_includes_current_key_and_call_counts(isolated_storage):
    first_key_id = isolated_storage.add_key("openrouter", "sk-alpha-1234")
    second_key_id = isolated_storage.add_key("openrouter", "sk-beta-5678")

    isolated_storage.record_key_usage(
        first_key_id,
        provider_id="openrouter-primary",
        model="openai/gpt-4o-mini",
        used_at="2026-05-16T11:00:00+00:00",
    )
    isolated_storage.record_key_usage(
        second_key_id,
        provider_id="openrouter-secondary",
        model="openai/gpt-4o",
        used_at="2026-05-16T11:05:00+00:00",
    )
    isolated_storage.record_key_usage(
        second_key_id,
        provider_id="openrouter-secondary",
        model="openai/gpt-4o",
        used_at="2026-05-16T11:06:00+00:00",
    )
    isolated_storage.record_key_exhausted(
        first_key_id,
        message="Quota exhausted for this key.",
        exhausted_at="2026-05-16T11:10:00+00:00",
    )

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    client = TestClient(app)

    keys_response = client.get("/admin/keys")
    summary_response = client.get("/admin/keys/summary")

    assert keys_response.status_code == 200
    assert summary_response.status_code == 200

    keys_payload = {item["id"]: item for item in keys_response.json()}
    summary_payload = summary_response.json()

    assert keys_payload[first_key_id]["status"] == "quota_exhausted"
    assert keys_payload[first_key_id]["last_status_message"] == "Quota exhausted for this key."
    assert keys_payload[second_key_id]["request_count"] == 2
    assert summary_payload["total_calls"] == 3
    assert summary_payload["quota_exhausted_keys"] == 1
    assert summary_payload["current_key"]["id"] == second_key_id


def test_storage_matches_generic_ollama_key_to_ollama_cloud_provider(isolated_storage):
    isolated_storage.add_key("ollama", "ollama-valid-key")

    matched = isolated_storage.get_keys_by_service("ollama_cloud")

    assert len(matched) == 1
    assert matched[0].service == "ollama"


def test_admin_test_key_resolves_generic_ollama_service(registry_backup, isolated_storage):
    key_id = isolated_storage.add_key("ollama", "ollama-valid-key")
    registry._active_instances = {
        "ollama-cloud-primary": SuccessProvider(
            id="ollama-cloud-primary",
            type="ollama_cloud",
            supported_models=["gpt-oss:120b"],
        )
    }

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    client = TestClient(app)

    response = client.post(f"/admin/keys/{key_id}/test")

    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert response.json()["message"] == "Key is active and valid!"


def test_admin_test_key_resolves_gemini_service(registry_backup, isolated_storage):
    key_id = isolated_storage.add_key("gemini", "gemini-valid-key")
    registry._active_instances = {
        "gemini-primary": SuccessProvider(
            id="gemini-primary",
            type="gemini",
            supported_models=["gemini-2.5-flash"],
        ),
    }

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    client = TestClient(app)

    response = client.post(f"/admin/keys/{key_id}/test")

    assert response.status_code == 200
    assert response.json()["status"] == "active"
