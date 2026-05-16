import pytest
import httpx

from backend.errors import ErrorType, GatewayError
from backend.providers.ollama_cloud import OllamaCloudProvider
from backend.providers.openrouter import OpenRouterProvider
from backend.providers.base import ProviderModel
from backend.providers.openai_compatible import ProviderCapabilities
from backend.schemas import ChatCompletionRequest


def test_ollama_convert_request_supports_tools_multimodal_and_reasoning():
    provider = OllamaCloudProvider(id="ollama-test", type="ollama_cloud")
    request = ChatCompletionRequest(
        model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image and then use the tool."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_price",
                            "arguments": '{"item":"hammer"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"price": 42}',
                "name": "lookup_price",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_price",
                    "parameters": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                        "required": ["item"],
                    },
                },
            }
        ],
        tool_choice="auto",
        response_format={"type": "json_object"},
        reasoning_effort="high",
        max_completion_tokens=256,
    )

    payload = provider.convert_request(request)

    assert payload["messages"][0]["images"] == ["ZmFrZQ=="]
    assert payload["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"item": "hammer"}
    assert payload["tools"] == request.tools
    assert payload["format"] == "json"
    assert payload["think"] == "high"
    assert payload["options"]["num_predict"] == 256


def test_ollama_rejects_audio_output_requests():
    provider = OllamaCloudProvider(id="ollama-test", type="ollama_cloud")
    request = ChatCompletionRequest(
        model="llama3.1",
        messages=[{"role": "user", "content": "hello"}],
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
    )

    with pytest.raises(GatewayError) as exc_info:
        provider.convert_request(request)

    assert exc_info.value.type == ErrorType.UNSUPPORTED_FEATURE
    assert "audio output" in exc_info.value.message


def test_ollama_headers_use_current_config_api_key():
    provider = OllamaCloudProvider(id="ollama-test", type="ollama_cloud")
    provider.config["api_key"] = "first-key"
    assert provider._get_headers()["Authorization"] == "Bearer first-key"

    provider.config["api_key"] = "second-key"
    assert provider._get_headers()["Authorization"] == "Bearer second-key"


@pytest.mark.asyncio
async def test_ollama_list_models_infers_capabilities_from_show_metadata(monkeypatch):
    original_async_client = httpx.AsyncClient
    show_responses = {
        "qwen3": {
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {"qwen3.context_length": 32768},
        },
        "gemma3": {
            "capabilities": ["completion", "vision"],
            "model_info": {"gemma3.context_length": 131072},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen3", "digest": "digest-qwen", "details": {"family": "qwen"}},
                        {"name": "gemma3", "digest": "digest-gemma", "details": {"family": "gemma"}},
                    ]
                },
            )

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)

    def build_client(*args, **kwargs):
        return original_async_client(transport=transport, *args, **kwargs)

    monkeypatch.setattr("backend.providers.ollama_cloud.httpx.AsyncClient", build_client)

    async def fake_fetch_show_data(self, client, model_id):
        return show_responses[model_id]

    monkeypatch.setattr(OllamaCloudProvider, "_fetch_show_data", fake_fetch_show_data)

    provider = OllamaCloudProvider(id="ollama-test", type="ollama_cloud", base_url="http://ollama.test")
    models = await provider.list_models()

    qwen_model = next(model for model in models if model.id == "qwen3")
    gemma_model = next(model for model in models if model.id == "gemma3")

    assert qwen_model.context_length == 32768
    assert qwen_model.capabilities.supports_tools is True
    assert qwen_model.capabilities.supports_parallel_tool_calls is True
    assert qwen_model.capabilities.supports_vision_input is False
    assert "tools" in qwen_model.supported_parameters

    assert gemma_model.context_length == 131072
    assert gemma_model.capabilities.supports_tools is False
    assert gemma_model.capabilities.supports_vision_input is True
    assert gemma_model.input_modalities == ["text", "image"]


def test_ollama_convert_request_uses_cached_model_capabilities():
    provider = OllamaCloudProvider(id="ollama-test", type="ollama_cloud")
    provider._models_cache = {
        "timestamp": 1.0,
        "models": [
            ProviderModel(
                id="gemma3",
                name="gemma3",
                capabilities=ProviderCapabilities(
                    supports_tools=False,
                    supports_tool_choice=False,
                    supports_parallel_tool_calls=False,
                    supports_vision_input=True,
                    supports_audio_input=False,
                    supports_audio_output=False,
                    supports_reasoning=True,
                    supports_response_format=True,
                ),
            )
        ],
    }

    request = ChatCompletionRequest(
        model="gemma3",
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

    with pytest.raises(GatewayError) as exc_info:
        provider.convert_request(request)

    assert exc_info.value.type == ErrorType.UNSUPPORTED_FEATURE
    assert "tool calling" in exc_info.value.message


def test_openrouter_convert_response_injects_provider_metadata():
    provider = OpenRouterProvider(id="openrouter-test", type="openrouter")
    response = provider.convert_response(
        {
            "id": "gen-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_price",
                                    "arguments": '{"item":"hammer"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        },
        model="router/default",
    )

    assert response.model == "router/default"
    assert response.provider.id == "openrouter-test"
    assert response.provider.actual_model == "openai/gpt-4o-mini"
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "lookup_price"


@pytest.mark.asyncio
async def test_openrouter_stream_chat_completion_uses_shared_stream_transport(monkeypatch):
    original_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer or-test-key"
        payload = request.read().decode("utf-8")
        assert '"stream":true' in payload
        return httpx.Response(
            200,
            content=(
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
                'data: [DONE]\n\n'
            ).encode(),
        )

    transport = httpx.MockTransport(handler)

    def build_client(*args, **kwargs):
        return original_async_client(transport=transport, *args, **kwargs)

    monkeypatch.setattr("backend.providers.openrouter.httpx.AsyncClient", build_client)

    provider = OpenRouterProvider(id="openrouter-test", type="openrouter", api_key="or-test-key")
    request = ChatCompletionRequest(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello."}],
    )

    chunks = []
    async for chunk in provider.stream_chat_completion(request):
        chunks.append(chunk.decode())

    body = "".join(chunks)
    assert '"content":"Hel"' in body
    assert body.endswith("data: [DONE]\n\n")


def test_ollama_convert_response_normalizes_tool_call_arguments():
    provider = OllamaCloudProvider(id="ollama-test", type="ollama_cloud")
    response = provider.convert_response(
        {
            "model": "llama3.1",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Need a lookup before answering.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_price",
                            "arguments": {"item": "hammer"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
            "prompt_eval_count": 7,
            "eval_count": 3,
        },
        model="alias/default",
    )

    tool_call = response.choices[0].message.tool_calls[0]
    assert response.choices[0].message.reasoning == "Need a lookup before answering."
    assert tool_call["type"] == "function"
    assert tool_call["function"]["arguments"] == '{"item": "hammer"}'
    assert response.usage.total_tokens == 10