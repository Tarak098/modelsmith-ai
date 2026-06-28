from fastapi import APIRouter, HTTPException
from auto_ai.app.core.models import SettingsResponse, SettingsUpdate
from auto_ai.app.infra.db import (
    get_setting, set_setting, get_setting_bool, get_setting_int, get_setting_float
)

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("", response_model=SettingsResponse)
def get_settings():
    """Retrieve application settings."""
    return {
        "gemini_api_key": get_setting("gemini_api_key"),
        "default_model": get_setting("default_model"),
        "enable_openml": get_setting_bool("enable_openml", True),
        "enable_kaggle": get_setting_bool("enable_kaggle", True),
        "enable_uci": get_setting_bool("enable_uci", True),
        "max_datasets": get_setting_int("max_datasets", 10),
        "cache_dir": get_setting("cache_dir") or "data/cache",
        "max_cache_size_mb": get_setting_int("max_cache_size_mb", 500),
        "timeout_sec": get_setting_int("timeout_sec", 30),
        "kaggle_username": get_setting("kaggle_username"),
        "kaggle_key": get_setting("kaggle_key"),
        "max_training_time": get_setting_int("max_training_time", 300),
        "max_project_time": get_setting_int("max_project_time", 1200),
        "max_retries": get_setting_int("max_retries", 5),
        "repository_priority": get_setting("repository_priority") or "cache,openml,kaggle,uci,sklearn",
        "dataset_score_threshold": get_setting_float("dataset_score_threshold", 0.4),
        "max_candidate_datasets": get_setting_int("max_candidate_datasets", 10),
        "cv_folds": get_setting_int("cv_folds", 3),
        "enable_automl_strategy": get_setting_bool("enable_automl_strategy", True),
        "enable_memory_reuse": get_setting_bool("enable_memory_reuse", True),
        "enable_data_leakage_detection": get_setting_bool("enable_data_leakage_detection", True)
    }

@router.post("", response_model=SettingsResponse)
def update_settings(update: SettingsUpdate):
    """Update application configuration."""
    if update.gemini_api_key is not None:
        set_setting("gemini_api_key", update.gemini_api_key)
    if update.default_model is not None:
        set_setting("default_model", update.default_model)
    if update.enable_openml is not None:
        set_setting("enable_openml", "true" if update.enable_openml else "false")
    if update.enable_kaggle is not None:
        set_setting("enable_kaggle", "true" if update.enable_kaggle else "false")
    if update.enable_uci is not None:
        set_setting("enable_uci", "true" if update.enable_uci else "false")
    if update.max_datasets is not None:
        set_setting("max_datasets", str(update.max_datasets))
    if update.cache_dir is not None:
        set_setting("cache_dir", update.cache_dir)
    if update.max_cache_size_mb is not None:
        set_setting("max_cache_size_mb", str(update.max_cache_size_mb))
    if update.timeout_sec is not None:
        set_setting("timeout_sec", str(update.timeout_sec))
    if update.kaggle_username is not None:
        set_setting("kaggle_username", update.kaggle_username)
    if update.kaggle_key is not None:
        set_setting("kaggle_key", update.kaggle_key)
    if update.max_training_time is not None:
        set_setting("max_training_time", str(update.max_training_time))
    if update.max_project_time is not None:
        set_setting("max_project_time", str(update.max_project_time))
    if update.max_retries is not None:
        set_setting("max_retries", str(update.max_retries))
    if update.repository_priority is not None:
        set_setting("repository_priority", update.repository_priority)
    if update.dataset_score_threshold is not None:
        set_setting("dataset_score_threshold", str(update.dataset_score_threshold))
    if update.max_candidate_datasets is not None:
        set_setting("max_candidate_datasets", str(update.max_candidate_datasets))
    if update.cv_folds is not None:
        set_setting("cv_folds", str(update.cv_folds))
    if update.enable_automl_strategy is not None:
        set_setting("enable_automl_strategy", "true" if update.enable_automl_strategy else "false")
    if update.enable_memory_reuse is not None:
        set_setting("enable_memory_reuse", "true" if update.enable_memory_reuse else "false")
    if update.enable_data_leakage_detection is not None:
        set_setting("enable_data_leakage_detection", "true" if update.enable_data_leakage_detection else "false")
        
    return get_settings()
