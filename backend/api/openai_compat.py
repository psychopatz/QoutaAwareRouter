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
            return p_id, provider.type, await provider.list_models()
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
                "owned_by": p_type
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
