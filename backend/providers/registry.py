from typing import Dict, Type, Optional
from .base import BaseProvider
from .gemini import GeminiProvider
from .ollama_local import OllamaLocalProvider
from .ollama_cloud import OllamaCloudProvider
from .openrouter import OpenRouterProvider

class ProviderRegistry:
    _instance = None
    _providers: Dict[str, Type[BaseProvider]] = {
        "gemini": GeminiProvider,
        "ollama_local": OllamaLocalProvider,
        "ollama_cloud": OllamaCloudProvider,
        "openrouter": OpenRouterProvider,
    }
    _active_instances: Dict[str, BaseProvider] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProviderRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, type_name: str, provider_class: Type[BaseProvider]):
        cls._providers[type_name] = provider_class

    def create_provider(self, **kwargs) -> BaseProvider:
        p_id = kwargs.get("id")
        type_name = kwargs.get("type")
        if not p_id or not type_name:
            raise ValueError("Provider config must include 'id' and 'type'")
        
        if type_name not in self._providers:
            raise ValueError(f"Unknown provider type: {type_name}")
        
        provider_class = self._providers[type_name]
        instance = provider_class(**kwargs)
        self._active_instances[p_id] = instance
        return instance

    def get_instance(self, provider_id: str) -> Optional[BaseProvider]:
        return self._active_instances.get(provider_id)

    def get_all_instances(self) -> Dict[str, BaseProvider]:
        return self._active_instances

registry = ProviderRegistry()
