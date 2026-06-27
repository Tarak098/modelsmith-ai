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

class SettingsResponse(BaseModel):
    gemini_api_key: str
    default_model: str
