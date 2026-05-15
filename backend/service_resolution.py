from typing import Dict, Iterable, Optional, Sequence

SERVICE_DEFINITIONS: Dict[str, Dict[str, Sequence[str] | str]] = {
    "ollama": {
        "aliases": ("ollama", "ollama_cloud"),
        "provider_types": ("ollama_cloud",),
        "label": "Ollama",
    },
    "openrouter": {
        "aliases": ("openrouter",),
        "provider_types": ("openrouter",),
        "label": "OpenRouter",
    },
    "openai": {
        "aliases": ("openai", "openai_compatible"),
        "provider_types": ("openai", "openai_compatible"),
        "label": "OpenAI Compatible",
    },
}


def normalize_service_name(service: Optional[str]) -> str:
    if not service:
        return ""

    lowered = service.strip().lower()
    for canonical, definition in SERVICE_DEFINITIONS.items():
        aliases = definition.get("aliases", ())
        if lowered == canonical or lowered in aliases:
            return canonical
    return lowered



def provider_types_for_service(service: Optional[str]) -> Sequence[str]:
    normalized = normalize_service_name(service)
    definition = SERVICE_DEFINITIONS.get(normalized)
    if not definition:
        return (normalized,) if normalized else ()
    provider_types = definition.get("provider_types", ())
    return tuple(provider_types) if provider_types else (normalized,)



def service_matches_provider(service: Optional[str], provider_type: Optional[str]) -> bool:
    if not service or not provider_type:
        return False
    return provider_type in provider_types_for_service(service)



def pick_provider_for_service(providers: Iterable, service: Optional[str]):
    provider_map = {provider.type: provider for provider in providers}
    for provider_type in provider_types_for_service(service):
        provider = provider_map.get(provider_type)
        if provider is not None:
            return provider
    return None



def service_label(service: Optional[str]) -> str:
    normalized = normalize_service_name(service)
    definition = SERVICE_DEFINITIONS.get(normalized)
    if definition and isinstance(definition.get("label"), str):
        return definition["label"]
    return (service or "").replace("_", " ").title()
