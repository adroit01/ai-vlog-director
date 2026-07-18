import os
import json
import logging
from typing import Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)

# Ensure directory is configured
DESCRIPTIONS_DIR = settings.DESCRIPTIONS_DIR

def get_video_description(file_name: str) -> Optional[Dict[str, Any]]:
    """Reads the video description file if it exists, parses the JSON, and returns it."""
    desc_path = DESCRIPTIONS_DIR / f"{file_name}.json"
    if desc_path.exists() and desc_path.stat().st_size > 0:
        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read/parse description file {desc_path}: {str(e)}", exc_info=True)
    return None

def save_video_description(file_name: str, description_text: str) -> bool:
    """Saves the video description JSON text to a file."""
    desc_path = DESCRIPTIONS_DIR / f"{file_name}.json"
    try:
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(description_text)
        logger.info(f"Successfully saved description for {file_name} to cache.")
        return True
    except Exception as e:
        logger.error(f"Failed to save description file {desc_path}: {str(e)}", exc_info=True)
        return False
