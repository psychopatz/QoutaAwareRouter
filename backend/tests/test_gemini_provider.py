import httpx
import pytest

from backend.providers.gemini import GeminiProvider
from backend.schemas import ChatCompletionRequest


@pytest.mark.asyncio
async def test_gemini_list_models_uses_native_models_api(monkeypatch):
	original_async_client = httpx.AsyncClient

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.method == "GET"
		assert request.url.path == "/v1beta/models"
		assert request.headers["x-goog-api-key"] == "gm-test-key"
		return httpx.Response(
			200,
			json={
				"models": [
					{
						"name": "models/gemini-2.5-flash",
						"baseModelId": "gemini-2.5-flash",
						"displayName": "Gemini 2.5 Flash",
						"description": "Fast multimodal Gemini model",
						"inputTokenLimit": 1048576,
						"outputTokenLimit": 8192,
						"thinking": True,
						"supportedGenerationMethods": ["generateContent"],
					},
					{
						"name": "models/gemini-embedding-001",
						"baseModelId": "gemini-embedding-001",
						"supportedGenerationMethods": ["embedContent"],
					},
				]
			},
		)

	transport = httpx.MockTransport(handler)

	def build_client(*args, **kwargs):
		return original_async_client(transport=transport)

	monkeypatch.setattr("backend.providers.gemini.httpx.AsyncClient", build_client)

	provider = GeminiProvider(id="gemini-primary", type="gemini", api_key="gm-test-key")
	models = await provider.list_models()

	assert len(models) == 1
	assert models[0].id == "gemini-2.5-flash"
	assert models[0].name == "Gemini 2.5 Flash"
	assert models[0].context_length == 1048576
	assert models[0].max_completion_tokens == 8192
	assert models[0].input_modalities == ["text", "image", "audio"]
	assert models[0].output_modalities == ["text"]
	assert "reasoning_effort" in models[0].supported_parameters
	assert models[0].capabilities is not None
	assert models[0].capabilities.supports_tools is True
	assert models[0].capabilities.supports_vision_input is True
	assert models[0].capabilities.supports_audio_input is True


@pytest.mark.asyncio
async def test_gemini_chat_completion_uses_openai_compat_endpoint(monkeypatch):
	original_async_client = httpx.AsyncClient

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.method == "POST"
		assert request.url.path == "/v1beta/openai/chat/completions"
		assert request.headers["authorization"] == "Bearer gm-test-key"
		payload = request.read().decode("utf-8")
		assert '"tool_choice":"auto"' in payload
		assert '"response_format":{"type":"json_object"}' in payload
		assert '"tools"' in payload
		return httpx.Response(
			200,
			json={
				"id": "chatcmpl-gemini-1",
				"object": "chat.completion",
				"created": 123,
				"model": "gemini-2.5-flash",
				"choices": [
					{
						"index": 0,
						"message": {"role": "assistant", "content": "{\"ok\":true}"},
						"finish_reason": "stop",
					}
				],
				"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
			},
		)

	transport = httpx.MockTransport(handler)

	def build_client(*args, **kwargs):
		return original_async_client(transport=transport)

	monkeypatch.setattr("backend.providers.gemini.httpx.AsyncClient", build_client)

	provider = GeminiProvider(id="gemini-primary", type="gemini", api_key="gm-test-key")
	response = await provider.chat_completion(
		ChatCompletionRequest(
			model="gemini-2.5-flash",
			messages=[
				{"role": "system", "content": "Return JSON only."},
				{"role": "user", "content": "Say ok."},
			],
			tools=[
				{
					"type": "function",
					"function": {
						"name": "lookup_status",
						"parameters": {"type": "object", "properties": {}},
					},
				}
			],
			tool_choice="auto",
			response_format={"type": "json_object"},
		)
	)

	assert response.model == "gemini-2.5-flash"
	assert response.provider is not None
	assert response.provider.id == "gemini-primary"
	assert response.provider.type == "gemini"
	assert response.provider.actual_model == "gemini-2.5-flash"
	assert response.choices[0].message.content == '{"ok":true}'


@pytest.mark.asyncio
async def test_gemini_chat_completion_with_google_options_uses_native_generate_content(monkeypatch):
	original_async_client = httpx.AsyncClient

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.method == "POST"
		assert request.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
		assert request.headers["x-goog-api-key"] == "gm-test-key"
		payload = request.read().decode("utf-8")
		assert '"cachedContent":"cachedContents/abc123"' in payload
		assert '"serviceTier":"flex"' in payload
		assert '"systemInstruction":{"parts":[{"text":"Return structured output."}]}' in payload
		assert '"functionDeclarations"' in payload
		assert '"functionCallingConfig":{"mode":"AUTO"}' in payload
		assert '"responseMimeType":"application/json"' in payload
		assert '"responseJsonSchema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}' in payload
		assert '"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"LOW"}' in payload
		return httpx.Response(
			200,
			json={
				"responseId": "gemini-native-1",
				"modelVersion": "gemini-2.5-flash-001",
				"candidates": [
					{
						"content": {
							"role": "model",
							"parts": [
								{"text": "Thinking through the schema.", "thought": True},
								{
									"functionCall": {
										"id": "call_1",
										"name": "lookup_status",
										"args": {"item": "router"},
									}
								},
							],
						},
						"finishReason": "STOP",
					}
				],
				"usageMetadata": {
					"promptTokenCount": 10,
					"candidatesTokenCount": 3,
					"totalTokenCount": 13,
				},
			},
		)

	transport = httpx.MockTransport(handler)

	def build_client(*args, **kwargs):
		return original_async_client(transport=transport)

	monkeypatch.setattr("backend.providers.gemini.httpx.AsyncClient", build_client)

	provider = GeminiProvider(id="gemini-primary", type="gemini", api_key="gm-test-key")
	response = await provider.chat_completion(
		ChatCompletionRequest(
			model="gemini-2.5-flash",
			messages=[
				{"role": "system", "content": "Return structured output."},
				{"role": "user", "content": "Check router status."},
			],
			tools=[
				{
					"type": "function",
					"function": {
						"name": "lookup_status",
						"parameters": {"type": "object", "properties": {"item": {"type": "string"}}},
					},
				}
			],
			tool_choice="auto",
			response_format={
				"type": "json_schema",
				"json_schema": {
					"schema": {
						"type": "object",
						"properties": {"ok": {"type": "boolean"}},
						"required": ["ok"],
					}
				},
			},
			extra_body={
				"google": {
					"cached_content": "cachedContents/abc123",
					"service_tier": "flex",
					"thinking_config": {"includeThoughts": True, "thinkingLevel": "LOW"},
				}
			},
		)
	)

	assert response.id == "gemini-native-1"
	assert response.provider.actual_model == "gemini-2.5-flash-001"
	assert response.choices[0].finish_reason == "tool_calls"
	assert response.choices[0].message.reasoning == "Thinking through the schema."
	assert response.choices[0].message.tool_calls[0]["id"] == "call_1"
	assert response.choices[0].message.tool_calls[0]["function"]["arguments"] == '{"item": "router"}'
	assert response.usage.total_tokens == 13


@pytest.mark.asyncio
async def test_gemini_stream_with_google_options_uses_native_stream_endpoint(monkeypatch):
	original_async_client = httpx.AsyncClient

	def handler(request: httpx.Request) -> httpx.Response:
		assert request.method == "POST"
		assert request.url.path == "/v1beta/models/gemini-2.5-flash:streamGenerateContent"
		assert request.url.query == b"alt=sse"
		assert request.headers["x-goog-api-key"] == "gm-test-key"
		return httpx.Response(
			200,
			content=(
				'data: {"responseId":"stream-native-1","candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n'
				'data: {"responseId":"stream-native-1","candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}\n\n'
			).encode(),
		)

	transport = httpx.MockTransport(handler)

	def build_client(*args, **kwargs):
		return original_async_client(transport=transport)

	monkeypatch.setattr("backend.providers.gemini.httpx.AsyncClient", build_client)

	provider = GeminiProvider(id="gemini-primary", type="gemini", api_key="gm-test-key")
	request = ChatCompletionRequest(
		model="gemini-2.5-flash",
		messages=[{"role": "user", "content": "Say hello."}],
		extra_body={"google": {"thinking_config": {"includeThoughts": True}}},
	)

	chunks = []
	async for chunk in provider.stream_chat_completion(request):
		chunks.append(chunk.decode())

	body = "".join(chunks)
	assert '"content":"Hel"' in body
	assert '"content":"lo"' in body
	assert '"finish_reason":"stop"' in body
	assert '"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}'.replace(" ", "") in body.replace(" ", "")
	assert body.endswith("data: [DONE]\n\n")