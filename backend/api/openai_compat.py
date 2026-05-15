from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..routing.router import Router
from ..providers.registry import registry
from ..errors import GatewayError
from ..logging_config import logger
import time

router = APIRouter()

import asyncio

_models_cache = {
    "timestamp": 0,
    "data": [],
    "by_service": {}
}
CACHE_TTL = 300  # 5 minutes


def invalidate_models_cache():
    global _models_cache
    _models_cache = {
        "timestamp": 0,
        "data": [],
        "by_service": {},
    }


def _set_provider_api_key(provider, api_key):
    if api_key:
        provider.config["api_key"] = api_key
    else:
        provider.config.pop("api_key", None)

    descriptor = getattr(type(provider), "api_key", None)
    if hasattr(provider, "api_key") and not isinstance(descriptor, property):
        provider.api_key = api_key


def _extract_pricing(raw: dict):
    pricing = raw.get("pricing")
    return pricing if isinstance(pricing, dict) else None


def _is_zero_cost(value) -> bool:
    if value in (None, "", "-1"):
        return False
    try:
        return Decimal(str(value)) == 0
    except (InvalidOperation, TypeError, ValueError):
        return False


def _is_free_model(model) -> bool:
    raw = model.raw if isinstance(model.raw, dict) else {}
    pricing = _extract_pricing(raw) or {}
    zero_cost_dimensions = [key for key, value in pricing.items() if _is_zero_cost(value)]
    if zero_cost_dimensions and all(_is_zero_cost(value) for value in pricing.values() if value not in (None, "", "-1")):
        return True

    model_id = (model.id or "").lower()
    model_name = (model.name or "").lower()
    return model_id.endswith(":free") or "(free)" in model_name


async def _list_provider_models(provider):
    from ..storage.sqlite import storage

    original_api_key = provider.config.get("api_key")
    active_keys = [key.key for key in storage.get_keys_by_service(provider.type) if key.status == "active"]
    candidate_api_keys = []
    if original_api_key:
        candidate_api_keys.append(original_api_key)
    candidate_api_keys.extend(key for key in active_keys if key and key != original_api_key)
    if not candidate_api_keys:
        candidate_api_keys = [None]

    last_error = None
    try:
        for index, candidate_api_key in enumerate(candidate_api_keys):
            _set_provider_api_key(provider, candidate_api_key)
            try:
                models = await provider.list_models()
                if models or index == len(candidate_api_keys) - 1:
                    return models
            except Exception as error:
                last_error = error
        if last_error:
            raise last_error
        return []
    finally:
        _set_provider_api_key(provider, original_api_key)

@router.get("/models")
@router.get("/models/{service_type}")
async def list_models(service_type: str = None):
    global _models_cache
    now = time.time()
    
    if now - _models_cache["timestamp"] < CACHE_TTL and _models_cache["data"]:
        if service_type:
            return {"object": "list", "data": _models_cache["by_service"].get(service_type, [])}
        return {"object": "list", "data": _models_cache["data"]}

    all_models = []
    by_service = {}
    providers = registry.get_all_instances()
    
    async def fetch_provider_models(p_id, provider):
        try:
            return p_id, provider.type, await _list_provider_models(provider)
        except Exception as e:
            logger.warning(f"Failed to list models for provider {p_id}: {e}")
            return p_id, provider.type, []
            
    tasks = [
        fetch_provider_models(p_id, provider)
        for p_id, provider in providers.items()
        if provider.enabled
    ]
    
    results = await asyncio.gather(*tasks) if tasks else []
    
    for p_id, p_type, models in results:
        for m in models:
            model_obj = {
                "id": f"{p_type}/{m.id}",
                "object": "model",
                "created": int(now),
                "owned_by": p_type,
                "name": m.name,
                "description": (m.raw or {}).get("description"),
                "context_length": m.context_length,
                "max_completion_tokens": m.max_completion_tokens,
                "input_modalities": m.input_modalities,
                "output_modalities": m.output_modalities,
                "supported_parameters": m.supported_parameters,
                "capabilities": m.capabilities.model_dump() if m.capabilities else None,
                "pricing": _extract_pricing(m.raw or {}),
                "is_free": _is_free_model(m),
            }
            all_models.append(model_obj)
            if p_type not in by_service:
                by_service[p_type] = []
            by_service[p_type].append(model_obj)
            
    _models_cache["timestamp"] = now
    _models_cache["data"] = all_models
    _models_cache["by_service"] = by_service
            
    if service_type:
        return {"object": "list", "data": by_service.get(service_type, [])}
    return {"object": "list", "data": all_models}

# Dependency injection for the router would be better, 
# but for now we'll use a global or pass it in main.py
_router_instance: Router = None

def set_router(router_instance: Router):
    global _router_instance
    _router_instance = router_instance

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not _router_instance:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    try:
        if request.stream:
            return await _router_instance.route_stream(request)
        
        response = await _router_instance.route(request)
        return response
    except GatewayError as e:
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    except Exception as e:
        logger.exception(f"Unhandled error in chat_completions: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": {"message": str(e), "type": "internal_error"}}
        )
