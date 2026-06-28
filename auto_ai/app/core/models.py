from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ProjectCreate(BaseModel):
    name: str = Field(..., description="Name of the research project")
    description: str = Field(..., description="Natural language request of the machine learning problem")
    uploaded_file_path: Optional[str] = Field(None, description="Path to uploaded file, if any")

class ProjectResume(BaseModel):
    project_id: str
    instruction: str = Field(..., description="E.g., 'Improve the model using CatBoost'")

class ProjectUpdate(BaseModel):
    status: Optional[str] = None
    category: Optional[str] = None
    error_message: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    category: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

class AgentTaskResponse(BaseModel):
    id: str
    project_id: str
    agent_name: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class LogResponse(BaseModel):
    id: int
    project_id: str
    agent_name: str
    level: str
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True

class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    default_model: Optional[str] = None
    enable_openml: Optional[bool] = None
    enable_kaggle: Optional[bool] = None
    enable_uci: Optional[bool] = None
    max_datasets: Optional[int] = None
    cache_dir: Optional[str] = None
    max_cache_size_mb: Optional[int] = None
    timeout_sec: Optional[int] = None
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    max_training_time: Optional[int] = None
    max_project_time: Optional[int] = None
    max_retries: Optional[int] = None
    repository_priority: Optional[str] = None
    dataset_score_threshold: Optional[float] = None
    max_candidate_datasets: Optional[int] = None
    cv_folds: Optional[int] = None
    enable_automl_strategy: Optional[bool] = None
    enable_memory_reuse: Optional[bool] = None
    enable_data_leakage_detection: Optional[bool] = None

class SettingsResponse(BaseModel):
    gemini_api_key: str
    default_model: str
    enable_openml: bool
    enable_kaggle: bool
    enable_uci: bool
    max_datasets: int
    cache_dir: str
    max_cache_size_mb: int
    timeout_sec: int
    kaggle_username: str
    kaggle_key: str
    max_training_time: int
    max_project_time: int
    max_retries: int
    repository_priority: str
    dataset_score_threshold: float
    max_candidate_datasets: int
    cv_folds: int
    enable_automl_strategy: bool
    enable_memory_reuse: bool
    enable_data_leakage_detection: bool
