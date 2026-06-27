import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "ModelSmith AI"
    DEBUG: bool = True
    
    # Path configuration
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    WORKSPACE_DIR: Path = Path(os.getenv("WORKSPACE_DIR", str(BASE_DIR.parent)))
    DATA_DIR: Path = WORKSPACE_DIR / "auto_ai" / "data"
    DB_PATH: Path = DATA_DIR / "modelsmith.db"
    
    # Gemini API settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # LLM Settings
    DEFAULT_MODEL: str = "gemini-2.5-flash"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Instantiate global settings
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "runs").mkdir(parents=True, exist_ok=True)
