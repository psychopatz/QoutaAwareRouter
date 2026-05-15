import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 7317
    DEBUG: bool = False

    # Config Path
    CONFIG_PATH: str = "config.yaml"

    # Security
    ADMIN_TOKEN: Optional[str] = os.getenv("QA_LLM_ROUTER_ADMIN_TOKEN")
    REDACT_AUTHORIZATION_HEADERS: bool = True
    LOG_PROMPTS: bool = False
    LOG_RESPONSES: bool = False

    # API Keys (Defaults can be overridden by env)
    OLLAMA_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    GOOGLE_AI_STUDIO_API_KEY: Optional[str] = None

settings = Settings()
