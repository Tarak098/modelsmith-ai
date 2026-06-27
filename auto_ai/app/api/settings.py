from fastapi import APIRouter, HTTPException
from auto_ai.app.core.models import SettingsResponse, SettingsUpdate
from auto_ai.app.infra.db import get_setting, set_setting

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("", response_model=SettingsResponse)
def get_settings():
    """Retrieve application settings."""
    api_key = get_setting("gemini_api_key")
    default_model = get_setting("default_model")
    return {
        "gemini_api_key": api_key,
        "default_model": default_model
    }

@router.post("", response_model=SettingsResponse)
def update_settings(update: SettingsUpdate):
    """Update application configuration."""
    if update.gemini_api_key is not None:
        set_setting("gemini_api_key", update.gemini_api_key)
    if update.default_model is not None:
        set_setting("default_model", update.default_model)
        
    api_key = get_setting("gemini_api_key")
    default_model = get_setting("default_model")
    
    return {
        "gemini_api_key": api_key,
        "default_model": default_model
    }
