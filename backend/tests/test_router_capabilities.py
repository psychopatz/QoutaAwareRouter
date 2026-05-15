import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.api import responses_compat
from backend.providers.base import BaseProvider, ProviderHealth, ProviderModel
from backend.providers.openai_compatible import ProviderCapabilities
from backend.providers.registry import registry
from backend.responses_schemas import ResponsesRequest, ResponsesResponse
from backend.routing.model_aliases import ModelAliasManager
from backend.routing.router import Router
from backend.schemas import ChatCompletionResponse, Choice, ProviderMetadata, ResponseMessage, Usage, ChatCompletionRequest
from backend.storage.responses import response_store
from backend.streaming.control import ProviderStreamControl


class FakeProvider(BaseProvider):
    def __init__(self, response_text: str, model_capabilities: ProviderCapabilities = None, **kwargs):
        super().__init__(**kwargs)
        self.response_text = response_text
        self.model_capabilities = model_capabilities
        self.calls = 0

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
        return [
            ProviderModel(
                id=self.supported_models[0],
                name=self.supported_models[0],
                capabilities=self.model_capabilities,
            )
        ]

    async def health_check(self):
        return ProviderHealth(healthy=True)

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        return ChatCompletionResponse(
            id=f"chatcmpl-{self.id}",
            created=123,
            model=request.model,
            provider=ProviderMetadata(id=self.id, type=self.type, actual_model=request.model),
            choices=[Choice(index=0, message=ResponseMessage(content=self.response_text), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream_chat_completion(self, request: ChatCompletionRequest, stream_control: ProviderStreamControl = None):
        yield b""

    def convert_request(self, openai_request: ChatCompletionRequest):
        return {}

    def convert_response(self, provider_response, model: str):
        raise NotImplementedError

    def convert_stream_chunk(self, provider_chunk: bytes):
        return provider_chunk

    def normalize_error(self, error):
        raise error



def test_responses_streaming_emits_multimodal_content_part_events(registry_backup):
    class FakeRouter:
        async def route(self, request: ChatCompletionRequest):
            raise AssertionError("non-stream route should not be called")

        async def iter_stream(self, request: ChatCompletionRequest, on_provider_selected=None, stream_control=None):
            if on_provider_selected:
                on_provider_selected(
                    {
                        "id": "openrouter-primary",
                        "type": "openrouter",
                        "actual_model": "openai/gpt-4o-audio",
                    }
                )

            async def generator():
                yield b'data: {"id":"chat_2","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{"reasoning":"Think","audio":{"data":"abc","transcript":"Hi"},"refusal":"Nope"},"finish_reason":null}]}\n\n'
                yield b'data: {"id":"chat_2","object":"chat.completion.chunk","created":123,"model":"default","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
                yield b'data: [DONE]\n\n'

            return generator()

    app = FastAPI()
    responses_compat.set_router(FakeRouter())
    app.include_router(responses_compat.router, prefix="/v1")
    client = TestClient(app)

    with client.stream("POST", "/v1/responses", json={"model": "default", "input": "hello", "stream": True}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: response.reasoning.delta" in body
    assert "event: response.reasoning.done" in body
    assert "event: response.output_audio.delta" in body
    assert "event: response.output_audio.done" in body
    assert "event: response.refusal.delta" in body
    assert "event: response.refusal.done" in body
    async def stream_chat_completion(self, request: ChatCompletionRequest, stream_control: ProviderStreamControl = None):
        yield b""

    def convert_request(self, openai_request: ChatCompletionRequest):
        return {}

    def convert_response(self, provider_response, model: str):
        raise NotImplementedError

    def convert_stream_chunk(self, provider_chunk: bytes):
        return provider_chunk

    def normalize_error(self, error):
        raise error


@pytest.fixture
def registry_backup():
    previous = dict(registry.get_all_instances())
    yield previous
    registry._active_instances = previous


@pytest.mark.asyncio
async def test_router_skips_unsupported_model_candidate(registry_backup):
    registry._active_instances = {
        "ollama-first": FakeProvider(
            id="ollama-first",
            type="ollama_local",
            supported_models=["llama3.1"],
            response_text="should not be used",
            model_capabilities=ProviderCapabilities(
                supports_tools=False,
                supports_tool_choice=False,
                supports_parallel_tool_calls=False,
                supports_vision_input=False,
                supports_audio_input=False,
                supports_audio_output=False,
                supports_reasoning=False,
                supports_response_format=False,
            ),
        ),
        "openrouter-second": FakeProvider(
            id="openrouter-second",
            type="openrouter",
            supported_models=["openai/gpt-4o-mini"],
            response_text="used fallback",
        ),
    }

    alias_manager = ModelAliasManager(
        {
            "default": {
                "candidates": [
                    {"provider": "ollama-first", "model": "llama3.1"},
                    {"provider": "openrouter-second", "model": "openai/gpt-4o-mini"},
                ]
            }
        }
    )
    router = Router(alias_manager, {})

    response = await router.route(
        ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "Use a tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_price",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )

    assert response.choices[0].message.content == "used fallback"
    assert registry._active_instances["ollama-first"].calls == 0
    assert registry._active_instances["openrouter-second"].calls == 1


def test_responses_endpoint_wraps_chat_completions(registry_backup):
    class FakeRouter:
        async def route(self, request: ChatCompletionRequest):
            assert request.messages[0].role == "system"
            assert request.messages[1].role == "user"
            return ChatCompletionResponse(
                id="resp-chat-1",
                created=123,
                model=request.model,
                provider=ProviderMetadata(id="openrouter-primary", type="openrouter", actual_model="openai/gpt-4o-mini"),
                choices=[
                    Choice(
                        index=0,
                        message=ResponseMessage(
                            content="Hello from responses",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup_price", "arguments": '{"item":"hammer"}'},
                                }
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

    app = FastAPI()
    responses_compat.set_router(FakeRouter())
    app.include_router(responses_compat.router, prefix="/v1")
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "default",
            "instructions": "You are terse.",
            "input": "Say hello.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["id"].startswith("resp_")
    assert payload["output_text"] == "Hello from responses"
    assert payload["output"][0]["type"] == "message"
    assert payload["output"][1]["type"] == "function_call"

    fetched = client.get(f"/v1/responses/{payload['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == payload["id"]


def test_responses_support_previous_response_id(registry_backup):
    class FakeRouter:
        def __init__(self):
            self.calls = 0

        async def route(self, request: ChatCompletionRequest):
            self.calls += 1
            if self.calls == 1:
                assert [message.role for message in request.messages] == ["user"]
                text = "First answer"
            else:
                assert [message.role for message in request.messages] == ["user", "assistant", "user"]
                assert request.messages[1].content == "First answer"
                text = "Second answer"

            return ChatCompletionResponse(
                id=f"resp-chat-{self.calls}",
                created=123 + self.calls,
                model=request.model,
                provider=ProviderMetadata(id="openrouter-primary", type="openrouter", actual_model="openai/gpt-4o-mini"),
                choices=[
                    Choice(
                        index=0,
                        message=ResponseMessage(content=text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

    app = FastAPI()
    responses_compat.set_router(FakeRouter())
    app.include_router(responses_compat.router, prefix="/v1")
    client = TestClient(app)

    first = client.post("/v1/responses", json={"model": "default", "input": "hello"})
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={"model": "default", "previous_response_id": first_id, "input": "and continue"},
    )

    assert second.status_code == 200
    assert second.json()["output_text"] == "Second answer"


def test_responses_streaming_translates_chat_stream(registry_backup):
    class FakeRouter:
        async def route(self, request: ChatCompletionRequest):
            raise AssertionError("non-stream route should not be called")

        async def iter_stream(self, request: ChatCompletionRequest, on_provider_selected=None, stream_control=None):
            if on_provider_selected:
                on_provider_selected(
                    {
                        "id": "openrouter-primary",
                        "type": "openrouter",
                        "actual_model": "openai/gpt-4o-mini",
                    }
                )

            async def generator():
                yield b'data: {"id":"chat_1","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
                yield b'data: {"id":"chat_1","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                yield b'data: {"id":"chat_1","object":"chat.completion.chunk","created":123,"model":"default","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
                yield b'data: [DONE]\n\n'

            return generator()

    app = FastAPI()
    responses_compat.set_router(FakeRouter())
    app.include_router(responses_compat.router, prefix="/v1")
    client = TestClient(app)

    with client.stream("POST", "/v1/responses", json={"model": "default", "input": "hello", "stream": True}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: response.created" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.content_part.added" in body
    assert "event: response.content_part.done" in body
    assert "event: response.completed" in body
    assert '"output_text": "Hello"' in body


def test_responses_cancel_endpoint_marks_pending_response(registry_backup):
    response_id = "resp_cancel_pending"
    response_store.create_pending(
        response=ResponsesResponse(
            id=response_id,
            created_at=123,
            status="in_progress",
            model="default",
        ),
        conversation_messages=[{"role": "user", "content": "hello"}],
        request={"model": "default", "input": "hello", "stream": True},
        stored=True,
    )

    app = FastAPI()
    responses_compat.set_router(object())
    app.include_router(responses_compat.router, prefix="/v1")
    client = TestClient(app)

    response = client.post(f"/v1/responses/{response_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    response_store.clear_cancel(response_id)


@pytest.mark.asyncio
async def test_responses_stream_emits_cancelled_event(registry_backup):
    cancel_called = asyncio.Event()

    class FakeRouter:
        async def route(self, request: ChatCompletionRequest):
            raise AssertionError("non-stream route should not be called")

        async def iter_stream(self, request: ChatCompletionRequest, on_provider_selected=None, stream_control=None):
            if on_provider_selected:
                on_provider_selected(
                    {
                        "id": "openrouter-primary",
                        "type": "openrouter",
                        "actual_model": "openai/gpt-4o-mini",
                    }
                )

            if stream_control is not None:
                async def cancel_hook():
                    cancel_called.set()

                stream_control.register_cancel_callback(cancel_hook, native_supported=True)

            async def generator():
                yield b'data: {"id":"chat_1","object":"chat.completion.chunk","created":123,"model":"default","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                await cancel_called.wait()
            return generator()

    responses_compat.set_router(FakeRouter())
    request = ResponsesRequest(model="default", input="hello", stream=True)
    chat_request = responses_compat.responses_request_to_chat_request(request)
    request_messages = [message.model_dump(exclude_none=True) for message in chat_request.messages]
    response_id = "resp_stream_cancel"

    stream = responses_compat._stream_responses(request, chat_request, request_messages, response_id)
    created = await anext(stream)
    in_progress = await anext(stream)
    response_store.request_cancel(response_id)
    remaining = [chunk async for chunk in stream]
    body = b"".join([created, in_progress, *remaining]).decode()

    assert cancel_called.is_set()
    assert "event: response.cancelled" in body
    assert '"provider_cancel_supported": true' in body
    response_store.clear_cancel(response_id)


def test_responses_endpoint_returns_not_found_for_missing_response(registry_backup):
    app = FastAPI()
    responses_compat.set_router(object())
    app.include_router(responses_compat.router, prefix="/v1")
    client = TestClient(app)

    response = client.get("/v1/responses/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"