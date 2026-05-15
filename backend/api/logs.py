from fastapi import APIRouter
from ..routing.traffic import traffic_manager

router = APIRouter()

@router.get("/traffic")
async def get_traffic():
    return traffic_manager.get_logs()
