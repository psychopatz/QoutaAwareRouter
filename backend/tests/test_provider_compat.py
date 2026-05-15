import pytest

from backend.errors import ErrorType, GatewayError
from backend.providers.ollama_cloud import OllamaCloudProvider
from backend.providers.openrouter import OpenRouterProvider
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