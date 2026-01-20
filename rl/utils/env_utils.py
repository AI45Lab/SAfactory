import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file once
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))


def get_env(key: str):
    """Get environment variable with logging."""
    value = os.environ.get(key)
    logger.info(f"ENV {key} = {value}")
    return value
