from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..routing.router import Router
from ..providers.registry import registry
from ..errors import GatewayError
from ..logging_config import logger
import time

router = APIRouter()

@router.get("/models")
async def list_models():
    all_models = []
    providers = registry.get_all_instances()
    
    for p_id, provider in providers.items():
        if not provider.enabled:
            continue
        try:
            models = await provider.list_models()
            for m in models:
                all_models.append({
                    "id": f"{provider.type}/{m.id}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider.type
                })
        except Exception as e:
            logger.warning(f"Failed to list models for provider {p_id}: {e}")
            
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
            # Phase 3: Streaming
            raise GatewayError("Streaming not yet implemented in Phase 1", status_code=501)
        
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
