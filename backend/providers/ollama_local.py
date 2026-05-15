from .ollama_cloud import OllamaCloudProvider

class OllamaLocalProvider(OllamaCloudProvider):
    """
    Ollama Local uses the same API as Ollama Cloud but typically without auth 
    and pointing to 127.0.0.1.
    """
    def __init__(self, **kwargs):
        if "base_url" not in kwargs:
            kwargs["base_url"] = "http://127.0.0.1:11434"
        super().__init__(**kwargs)

    def _get_headers(self) -> dict:
        # Local usually doesn't need auth, but we support it if provided
        return super()._get_headers()
