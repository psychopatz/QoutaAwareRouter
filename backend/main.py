import uvicorn
from fastapi import FastAPI
from .settings import settings
from .logging_config import setup_logging
from .config_loader import ConfigLoader
from .routing.model_aliases import ModelAliasManager
from .routing.router import Router
from .api import health, openai_compat, responses_compat, admin, logs

app = FastAPI(title="Quota Aware LLM Router", version="0.1.0")

def init_app():
    setup_logging()
    
    # Load Config
    config_loader = ConfigLoader(settings.CONFIG_PATH)
    config = config_loader.load()
    
    # Setup Routing
    alias_manager = ModelAliasManager(config_loader.get_model_aliases())
    router_instance = Router(alias_manager, config_loader.get_routing_config())
    
    # Inject Router into API modules
    openai_compat.set_router(router_instance)
    responses_compat.set_router(router_instance)
    
    # Include Routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(openai_compat.router, prefix="/v1", tags=["OpenAI Compatibility"])
    app.include_router(responses_compat.router, prefix="/v1", tags=["OpenAI Compatibility"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"])
    app.include_router(logs.router, prefix="/admin", tags=["Admin"])

init_app()

if __name__ == "__main__":
    uvicorn.run(
        "quota_aware_llm_router.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
