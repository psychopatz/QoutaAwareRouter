from fastapi import APIRouter, HTTPException
from ..storage.sqlite import storage, ApiKey
from pydantic import BaseModel

router = APIRouter()

class KeyCreate(BaseModel):
    service: str
    key: str

@router.get("/keys")
async def list_keys():
    return storage.get_all_keys()

@router.post("/keys")
async def add_key(data: KeyCreate):
    key_id = storage.add_key(data.service, data.key)
    return {"id": key_id, "status": "added"}

@router.delete("/keys/{key_id}")
async def delete_key(key_id: int):
    storage.delete_key(key_id)
    return {"status": "deleted"}

@router.post("/keys/{key_id}/status")
async def update_status(key_id: int, status: str):
    storage.update_key_status(key_id, status)
    return {"status": "updated"}
