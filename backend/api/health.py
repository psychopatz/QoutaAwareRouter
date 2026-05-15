from fastapi import APIRouter
from ..providers.registry import registry

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/providers")
async def list_providers():
    providers = registry.get_all_instances()
    return {
        "providers": [
            {
                "id": p.id,
                "type": p.type,
                "enabled": p.enabled,
                "priority": p.priority
            } for p in providers.values()
        ]
    }

@router.get("/providers/{provider_id}/health")
async def provider_health(provider_id: str):
    provider = registry.get_instance(provider_id)
    if not provider:
        return {"error": "Provider not found"}, 404
    
    health = await provider.health_check()
    return health.model_dump()
