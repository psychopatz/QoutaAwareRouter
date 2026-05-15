from fastapi import APIRouter, HTTPException
from ..storage.sqlite import storage, ApiKey
from ..service_resolution import pick_provider_for_service
from pydantic import BaseModel
from .openai_compat import invalidate_models_cache

router = APIRouter()

class KeyCreate(BaseModel):
    service: str
    key: str

@router.get("/keys")
async def list_keys():
    return storage.get_all_keys()

@router.get("/keys/summary")
async def key_summary():
    return storage.get_key_summary()

@router.post("/keys")
async def add_key(data: KeyCreate):
    key_id = storage.add_key(data.service, data.key)
    invalidate_models_cache()
    return {"id": key_id, "status": "added"}

@router.delete("/keys/{key_id}")
async def delete_key(key_id: int):
    storage.delete_key(key_id)
    invalidate_models_cache()
    return {"status": "deleted"}

@router.post("/keys/{key_id}/status")
async def update_status(key_id: int, status: str):
    storage.update_key_status(key_id, status)
    invalidate_models_cache()
    return {"status": "updated"}

@router.post("/keys/{key_id}/test")
async def test_key(key_id: int):
    keys = storage.get_all_keys()
    key = next((k for k in keys if k.id == key_id), None)
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
        
    from ..providers.registry import registry
    target_provider = pick_provider_for_service(registry.get_all_instances().values(), key.service)
            
    if not target_provider:
        raise HTTPException(status_code=400, detail=f"No provider instance found for service type: {key.service}")
        
    is_valid = await target_provider.test_key(key.key)
    
    new_status = "active" if is_valid else "auth_failed"
    storage.update_key_status(
        key_id,
        new_status,
        last_status_message=None if is_valid else "Authentication failed or key is invalid/exhausted.",
    )
    invalidate_models_cache()
    
    return {
        "status": new_status,
        "message": "Key is active and valid!" if is_valid else "Authentication failed or key is invalid/exhausted."
    }
