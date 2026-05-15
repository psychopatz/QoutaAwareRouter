import yaml
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from .logging_config import logger
from .providers.registry import registry

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

        self._initialize_providers()
        return self.config_data

    def _initialize_providers(self):
        provider_configs = self.config_data.get("providers", [])
        for p_cfg in provider_configs:
            try:
                p_id = p_cfg.get("id")
                p_type = p_cfg.get("type")
                if not p_id or not p_type:
                    logger.error(f"Provider config missing id or type: {p_cfg}")
                    continue
                
                registry.create_provider(**p_cfg)
                logger.info(f"Initialized provider: {p_id} ({p_type})")
            except Exception as e:
                logger.error(f"Failed to initialize provider {p_cfg.get('id')}: {e}")

    def get_model_aliases(self) -> Dict[str, Any]:
        return self.config_data.get("model_aliases", {})

    def get_routing_config(self) -> Dict[str, Any]:
        return self.config_data.get("routing", {})

    def get_server_config(self) -> Dict[str, Any]:
        return self.config_data.get("server", {})
