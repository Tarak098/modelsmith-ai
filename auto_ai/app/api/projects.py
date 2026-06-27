import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from auto_ai.app.core.models import ProjectResponse, AgentTaskResponse, LogResponse
from auto_ai.app.core.security import sanitize_filename
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import (
    create_project_record, update_project_record, get_project_record,
    list_projects_records, get_agent_tasks_records, get_execution_logs_records
)
from auto_ai.app.services.coordinator import CoordinatorAgent
from auto_ai.app.services.memory_agent import MemoryAgent

router = APIRouter(prefix="/projects", tags=["projects"])
coordinator = CoordinatorAgent()
memory_agent = MemoryAgent()

@router.get("", response_model=List[ProjectResponse])
def list_projects():
    """List all research projects."""
    return list_projects_records()

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str):
    """Retrieve detailed project status."""
    project = get_project_record(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.post("")
def create_project(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Initialize a new machine learning research project and run coordinator in the background.
    """
    project_id = str(uuid.uuid4())
    uploaded_file_path = None
    
    # 1. Save uploaded file if any
    if file:
        try:
            sanitized_name = sanitize_filename(file.filename)
            contents = file.file.read()
            saved_path = StorageManager.save_uploaded_file(contents, f"{project_id}_{sanitized_name}")
            uploaded_file_path = str(saved_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process file upload: {str(e)}")
            
    # 2. Check Memory for context continuation
    # If description represents an expansion, we merge past memory context
    similar = memory_agent.search_similar_project(description)
    enhanced_desc = description
    if similar:
        prev_model = similar.get("summary", {}).get("model_name", "Unknown")
        prev_metrics = similar.get("summary", {}).get("metrics", {})
        enhanced_desc = (
            f"{description}. [Memory Context: This request is related to previous project ID: {similar['project_id']} "
            f"('{similar['name']}') which used a '{prev_model}' with metrics {prev_metrics}]."
        )
        
    # 3. Create database records
    project = create_project_record(project_id, name, enhanced_desc)
    
    # 4. Trigger Coordinator workflow in a background task thread
    background_tasks.add_task(
        coordinator.run_workflow,
        project_id=project_id,
        name=name,
        description=enhanced_desc,
        uploaded_file_path=uploaded_file_path
    )
    
    return {"project_id": project_id, "status": "running"}

@router.get("/{project_id}/tasks", response_model=List[AgentTaskResponse])
def get_project_tasks(project_id: str):
    """Get the list of agent task statuses for a project."""
    return get_agent_tasks_records(project_id)

@router.get("/{project_id}/logs", response_model=List[LogResponse])
def get_project_logs(project_id: str):
    """Retrieve execution log telemetry for the dashboard."""
    return get_execution_logs_records(project_id)

@router.get("/{project_id}/report")
def get_project_report(project_id: str):
    """Serve the final HTML report file to render in an iframe."""
    run_dir = StorageManager.get_run_dir(project_id)
    report_path = run_dir / "reports" / "report.html"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet or project failed.")
        
    return FileResponse(report_path)
