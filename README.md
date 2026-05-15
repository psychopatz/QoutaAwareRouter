# Quota Aware LLM Router

A lightweight, modular, provider-agnostic LLM router that exposes an OpenAI-compatible API.

## Features
- **Provider Agnostic**: Support for Ollama (Local/Cloud), OpenRouter, Google AI Studio, and generic OpenAI-compatible services.
- **Quota Aware**: Respects provider rate limits, quotas, and cooldowns.
- **Priority Routing**: Ordered candidate failover for reliable inference.
- **OpenAI Compatible**: Drop-in replacement for OpenAI SDKs and tools.
- **Lightweight**: Minimal dependencies, local configuration.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Copy `.env.example` to `.env` and fill in your API keys:
    ```bash
    cp .env.example .env
    ```
4.  Configure your providers and model aliases in `config.yaml`.

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
```

## Project Structure
- `quota_aware_llm_router/api/`: OpenAI-compatible and management endpoints.
- `quota_aware_llm_router/providers/`: Provider adapters (Ollama, OpenRouter, etc.).
- `quota_aware_llm_router/routing/`: Routing logic and failover policies.
- `quota_aware_llm_router/schemas.py`: OpenAI-style Pydantic models.
