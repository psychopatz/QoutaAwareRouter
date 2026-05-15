import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from cryptography.fernet import Fernet

from ..service_resolution import normalize_service_name, service_matches_provider

DB_PATH = "qarouter.db"
KEY_FILE = "encryption_key.key"

class ApiKey(BaseModel):
    id: Optional[int] = None
    service: str  # e.g., "ollama", "openrouter"
    key: str
    status: str = "active"  # active, rate_limited, quota_exhausted, auth_failed
    last_used: Optional[str] = None
    cooldown_until: Optional[str] = None
    request_count: int = 0
    last_used_provider_id: Optional[str] = None
    last_used_model: Optional[str] = None
    last_status_message: Optional[str] = None
    exhausted_at: Optional[str] = None

class Storage:
    def __init__(self, db_path: str = DB_PATH, key_file: str = KEY_FILE):
        self.db_path = db_path
        self.key_file = key_file
        self.cipher = self._init_cipher()
        self._init_db()

    def _init_cipher(self) -> Fernet:
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
        return Fernet(key)

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _ensure_column(self, cursor: sqlite3.Cursor, table_name: str, column_name: str, definition: str):
        columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _encrypt(self, text: str) -> str:
        return self.cipher.encrypt(text.encode()).decode()

    def _decrypt(self, text: str) -> str:
        try:
            return self.cipher.decrypt(text.encode()).decode()
        except:
            return text

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    key TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    last_used TIMESTAMP,
                    cooldown_until TIMESTAMP,
                    request_count INTEGER DEFAULT 0,
                    last_used_provider_id TEXT,
                    last_used_model TEXT,
                    last_status_message TEXT,
                    exhausted_at TIMESTAMP
                )
            """)
            self._ensure_column(cursor, "api_keys", "request_count", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "api_keys", "last_used_provider_id", "TEXT")
            self._ensure_column(cursor, "api_keys", "last_used_model", "TEXT")
            self._ensure_column(cursor, "api_keys", "last_status_message", "TEXT")
            self._ensure_column(cursor, "api_keys", "exhausted_at", "TIMESTAMP")
            conn.commit()

    def add_key(self, service: str, key: str) -> int:
        encrypted_key = self._encrypt(key)
        normalized_service = normalize_service_name(service)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO api_keys (service, key) VALUES (?, ?)",
                (normalized_service, encrypted_key)
            )
            return cursor.lastrowid

    def get_keys_by_service(self, service: str) -> List[ApiKey]:
        keys = self.get_all_keys()
        return [key for key in keys if service_matches_provider(key.service, service)]

    def update_key_status(
        self,
        key_id: int,
        status: str,
        cooldown_until: Optional[str] = None,
        last_status_message: Optional[str] = None,
        exhausted_at: Optional[str] = None,
    ):
        if status == "active":
            last_status_message = None
            exhausted_at = None
            cooldown_until = None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE api_keys
                SET status = ?, cooldown_until = ?, last_status_message = ?, exhausted_at = ?
                WHERE id = ?
                """,
                (status, cooldown_until, last_status_message, exhausted_at, key_id)
            )

    def record_key_usage(self, key_id: int, provider_id: str, model: str, used_at: Optional[str] = None):
        timestamp = used_at or self._utcnow()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE api_keys
                SET last_used = ?,
                    last_used_provider_id = ?,
                    last_used_model = ?,
                    request_count = COALESCE(request_count, 0) + 1
                WHERE id = ?
                """,
                (timestamp, provider_id, model, key_id),
            )

    def record_key_exhausted(self, key_id: int, message: str, exhausted_at: Optional[str] = None):
        self.update_key_status(
            key_id,
            "quota_exhausted",
            last_status_message=message,
            exhausted_at=exhausted_at or self._utcnow(),
        )

    def get_key_summary(self) -> Dict[str, Any]:
        keys = self.get_all_keys()
        current_key = None
        if keys:
            current_key = max(keys, key=lambda key: (key.last_used or "", key.request_count, key.id or 0))

        return {
            "total_keys": len(keys),
            "active_keys": sum(1 for key in keys if key.status == "active"),
            "inactive_keys": sum(1 for key in keys if key.status != "active"),
            "quota_exhausted_keys": sum(1 for key in keys if key.status == "quota_exhausted"),
            "total_calls": sum(key.request_count for key in keys),
            "current_key": current_key.model_dump() if current_key else None,
        }

    def delete_key(self, key_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))

    def get_all_keys(self) -> List[ApiKey]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys")
            rows = cursor.fetchall()
            keys = []
            for row in rows:
                d = dict(row)
                d['service'] = normalize_service_name(d['service'])
                d['key'] = self._decrypt(d['key'])
                keys.append(ApiKey(**d))
            return keys

storage = Storage()
