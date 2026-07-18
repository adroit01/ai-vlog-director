import yaml
from pathlib import Path
from typing import Dict, Any

PROMPTS_PATH = Path(__file__).resolve().parent / "agent_prompts.yaml"

_prompts_cache: Dict[str, Any] = {}

def load_prompts() -> Dict[str, Any]:
    """Loads and caches the prompt templates from the YAML file."""
    global _prompts_cache
    if not _prompts_cache:
        if not PROMPTS_PATH.exists():
            raise FileNotFoundError(f"Prompts configuration file not found at {PROMPTS_PATH}")
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            _prompts_cache = data.get("prompts", {})
    return _prompts_cache

def get_prompt_resource(name: str) -> str:
    """Retrieves the text of a prompt template by its key name from agent_prompts.yaml."""
    prompts = load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt resource '{name}' not found. Available prompts: {list(prompts.keys())}")
    return prompts[name]["text"]
