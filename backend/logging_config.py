import logging
import sys
from .settings import settings

def setup_logging():
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Suppress some noisy logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("pydantic").setLevel(logging.INFO)

logger = logging.getLogger("qarouter")
