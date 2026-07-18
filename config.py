import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
OUTPUT_DIR = STORAGE_DIR / "outputs"
TEMP_DIR = STORAGE_DIR / "temp"
DESCRIPTIONS_DIR = STORAGE_DIR / "descriptions"

# Create directories if they do not exist
for directory in [STORAGE_DIR, UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, DESCRIPTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Langfuse configuration
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # F5-TTS Server configuration
    F5_TTS_HOST: str = os.getenv("F5_TTS_HOST", "http://localhost:8000")
    
    # Database URL
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR}/vlogs.db")
    
    # Ollama Host URL
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Directory paths
    STORAGE_DIR = STORAGE_DIR
    UPLOAD_DIR = UPLOAD_DIR
    OUTPUT_DIR = OUTPUT_DIR
    TEMP_DIR = TEMP_DIR
    DESCRIPTIONS_DIR = DESCRIPTIONS_DIR

settings = Settings()
