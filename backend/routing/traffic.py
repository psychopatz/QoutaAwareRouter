from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time

class TrafficLog(BaseModel):
    id: str
    timestamp: float
    method: str
    path: str
    model: str
    provider_id: Optional[str] = None
    status_code: int
    latency_ms: float
    error: Optional[str] = None

TrafficLog.model_rebuild()

class TrafficManager:
    def __init__(self, max_logs: int = 100):
        self.logs: List[TrafficLog] = []
        self.max_logs = max_logs

    def add_log(self, log: TrafficLog):
        self.logs.insert(0, log)
        if len(self.logs) > self.max_logs:
            self.logs.pop()

    def get_logs(self) -> List[TrafficLog]:
        return self.logs

traffic_manager = TrafficManager()
