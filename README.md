# Quota Aware LLM Router

A lightweight, modular, provider-agnostic LLM router that exposes an OpenAI-compatible API.

## Features

- **Provider Agnostic**: Support for Ollama (Local/Cloud), OpenRouter, Google AI Studio, and generic OpenAI-compatible services.
- **Quota Aware**: Respects provider rate limits, quotas, and cooldowns.
- **Priority Routing**: Ordered candidate failover for reliable inference.
- **OpenAI Compatible**: Drop-in replacement for OpenAI chat completions plus a gated Responses API surface.
- **Lightweight**: Minimal dependencies, local configuration.

## Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

4. Configure your providers and model aliases in `config.yaml`.

## Running the Server

```bash
uvicorn quota_aware_llm_router.main:app --host 127.0.0.1 --port 7317 --reload
```

## Quick Start (cURL)

```bash
curl http://127.0.0.1:7317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "stream": false
  }'
```

## Configuration

Edit `config.yaml` to define your providers and model aliases.

```yaml
providers:
  - id: ollama-local-main
    type: ollama_local
    enabled: true
    base_url: "http://127.0.0.1:11434"
    priority: 10
    models:
      - "llama3.1"

model_aliases:
  default:
    candidates:
      - provider: ollama-local-main
        model: llama3.1

# Optional per-model overrides for cases where a provider knows less than the model.
# Values override discovered metadata for this provider/model.
# model_capabilities:
#   llama3.1:
#     supports_tools: true
#     supports_vision_input: true
```

OpenRouter model metadata is discovered from its `/models` API and used to pre-filter candidates for tools, vision, audio, reasoning, and structured outputs before a request is sent. Ollama uses provider defaults plus optional `model_capabilities` overrides in `config.yaml`.

## Responses API

OpenAI-style Responses API requests are available at `/v1/responses`, including named SSE streaming events, response retrieval, cancellation, and `previous_response_id` conversation chaining for stored responses.

```bash
curl http://127.0.0.1:7317/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Summarize the latest router state in one sentence."
  }'
```

Streaming clients can send `"stream": true` and receive Responses-style SSE events such as `response.created`, `response.output_text.delta`, `response.output_item.done`, `response.completed`, and `response.cancelled`.

The bridge now also emits broader content-part events for richer outputs, including `response.content_part.added`, `response.content_part.done`, `response.reasoning.delta`, `response.reasoning.done`, `response.output_audio.delta`, `response.output_audio.done`, `response.refusal.delta`, and `response.refusal.done` when the upstream provider returns those deltas.

Stored responses can be fetched with `GET /v1/responses/{response_id}` and cancelled in-flight with `POST /v1/responses/{response_id}/cancel`. Follow-up turns can send `previous_response_id` to reuse the stored conversation state without rebuilding the full prior message history client-side.

For provider-side cancellation, OpenRouter streaming requests now use upstream abort semantics when a response is cancelled. Ollama streaming requests use a best-effort transport close, but Ollama does not currently expose a documented dedicated generation-cancel endpoint for `/api/chat`.

## Project Structure

- `quota_aware_llm_router/api/`: OpenAI-compatible and management endpoints.
- `quota_aware_llm_router/providers/`: Provider adapters (Ollama, OpenRouter, etc.).
- `quota_aware_llm_router/routing/`: Routing logic and failover policies.
- `quota_aware_llm_router/schemas.py`: OpenAI-style Pydantic models.
