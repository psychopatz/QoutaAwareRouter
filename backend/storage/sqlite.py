import sqlite3
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from cryptography.fernet import Fernet

DB_PATH = "qarouter.db"
KEY_FILE = "encryption_key.key"

class ApiKey(BaseModel):
    id: Optional[int] = None
    service: str  # e.g., "ollama", "openrouter"
    key: str
    status: str = "active"  # active, rate_limited, auth_failed
    last_used: Optional[str] = None
    cooldown_until: Optional[str] = None

class Storage:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.cipher = self._init_cipher()
        self._init_db()

    def _init_cipher(self) -> Fernet:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as f:
                f.write(key)
        return Fernet(key)

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
                    cooldown_until TIMESTAMP
                )
            """)
            conn.commit()

    def add_key(self, service: str, key: str) -> int:
        encrypted_key = self._encrypt(key)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO api_keys (service, key) VALUES (?, ?)",
                (service, encrypted_key)
            )
            return cursor.lastrowid

    def get_keys_by_service(self, service: str) -> List[ApiKey]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys WHERE service = ?", (service,))
            rows = cursor.fetchall()
            keys = []
            for row in rows:
                d = dict(row)
                d['key'] = self._decrypt(d['key'])
                keys.append(ApiKey(**d))
            return keys

    def update_key_status(self, key_id: int, status: str, cooldown_until: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET status = ?, cooldown_until = ? WHERE id = ?",
                (status, cooldown_until, key_id)
            )

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
                d['key'] = self._decrypt(d['key'])
                keys.append(ApiKey(**d))
            return keys

storage = Storage()
