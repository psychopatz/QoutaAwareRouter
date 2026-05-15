import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..responses_schemas import ResponsesResponse
from .sqlite import DB_PATH


class StoredResponse(BaseModel):
    response: ResponsesResponse
    conversation_messages: List[Dict[str, Any]] = Field(default_factory=list)
    request: Dict[str, Any] = Field(default_factory=dict)
    stored: bool = True


class ResponseStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cancel_requests = set()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    conversation_messages_json TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    stored INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()

    def upsert(self, stored_response: StoredResponse):
        response_payload = stored_response.response.model_dump(exclude_none=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO responses (
                    id,
                    status,
                    model,
                    response_json,
                    conversation_messages_json,
                    request_json,
                    stored
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    model = excluded.model,
                    response_json = excluded.response_json,
                    conversation_messages_json = excluded.conversation_messages_json,
                    request_json = excluded.request_json,
                    stored = excluded.stored
                """,
                (
                    stored_response.response.id,
                    stored_response.response.status,
                    stored_response.response.model,
                    json.dumps(response_payload),
                    json.dumps(stored_response.conversation_messages),
                    json.dumps(stored_response.request),
                    1 if stored_response.stored else 0,
                ),
            )
            conn.commit()

    def create_pending(
        self,
        response: ResponsesResponse,
        conversation_messages: List[Dict[str, Any]],
        request: Dict[str, Any],
        stored: bool = True,
    ) -> StoredResponse:
        pending = StoredResponse(
            response=response,
            conversation_messages=conversation_messages,
            request=request,
            stored=stored,
        )
        self.upsert(pending)
        return pending

    def get(self, response_id: str, include_ephemeral: bool = False) -> Optional[StoredResponse]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM responses WHERE id = ?", (response_id,))
            row = cursor.fetchone()

        if not row:
            return None

        stored = bool(row["stored"])
        if not stored and not include_ephemeral:
            return None

        return StoredResponse(
            response=ResponsesResponse(**json.loads(row["response_json"])),
            conversation_messages=json.loads(row["conversation_messages_json"]),
            request=json.loads(row["request_json"]),
            stored=stored,
        )

    def request_cancel(self, response_id: str) -> Optional[StoredResponse]:
        stored_response = self.get(response_id, include_ephemeral=True)
        if not stored_response:
            return None

        if stored_response.response.status not in {"completed", "failed", "cancelled"}:
            with self._lock:
                self._cancel_requests.add(response_id)
            stored_response = stored_response.model_copy(
                update={
                    "response": stored_response.response.model_copy(update={"status": "cancelling"})
                }
            )
            self.upsert(stored_response)

        return stored_response

    def is_cancel_requested(self, response_id: str) -> bool:
        with self._lock:
            return response_id in self._cancel_requests

    def clear_cancel(self, response_id: str):
        with self._lock:
            self._cancel_requests.discard(response_id)


response_store = ResponseStore()