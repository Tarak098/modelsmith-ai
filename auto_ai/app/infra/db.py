import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from auto_ai.app.config import settings
from auto_ai.app.utils.logging import logger

def get_db_connection() -> sqlite3.Connection:
    """Return a standard connection to the SQLite database."""
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables in SQLite database if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Projects Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            category TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            error_message TEXT
        )
    """)
    
    # 2. Agent Tasks Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_tasks (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            output_data TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)
    
    # 3. Execution Logs Table (created in logging.py too, but here for completeness)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    """)
    
    # 4. Long Term Memory Store Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_store (
            id TEXT PRIMARY KEY,
            memory_key TEXT NOT NULL,
            memory_value TEXT NOT NULL,
            created_at TIMESTAMP
        )
    """)
    
    # 5. Application Settings Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL
        )
    """)
    
    # Set default settings if not already present
    cursor.execute("SELECT COUNT(*) FROM settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO settings (setting_key, setting_value) VALUES (?, ?)",
            ("gemini_api_key", settings.GEMINI_API_KEY)
        )
        cursor.execute(
            "INSERT INTO settings (setting_key, setting_value) VALUES (?, ?)",
            ("default_model", settings.DEFAULT_MODEL)
        )
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# Project Helpers
def create_project_record(project_id: str, name: str, description: str) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO projects (id, name, description, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, name, description, "pending", now, now)
    )
    conn.commit()
    conn.close()
    return {
        "id": project_id,
        "name": name,
        "description": description,
        "status": "pending",
        "category": None,
        "created_at": now,
        "updated_at": now,
        "error_message": None
    }

def update_project_record(project_id: str, status: Optional[str] = None, category: Optional[str] = None, error_message: Optional[str] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    updates = []
    params = []
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if category is not None:
        updates.append("category = ?")
        params.append(category)
    if error_message is not None:
        updates.append("error_message = ?")
        params.append(error_message)
        
    updates.append("updated_at = ?")
    params.append(now)
    
    params.append(project_id)
    query = f"UPDATE projects SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, tuple(params))
    conn.commit()
    conn.close()

def get_project_record(project_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_projects_records() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Agent Task Helpers
def save_agent_task_record(project_id: str, agent_name: str, status: str, started_at: Optional[str] = None, completed_at: Optional[str] = None, output_data: Optional[Dict[str, Any]] = None):
    task_id = f"{project_id}:{agent_name}"
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if task already exists
    cursor.execute("SELECT COUNT(*) FROM agent_tasks WHERE id = ?", (task_id,))
    exists = cursor.fetchone()[0] > 0
    
    output_str = json.dumps(output_data) if output_data is not None else None
    
    if exists:
        updates = ["status = ?"]
        params = [status]
        if completed_at:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if output_str:
            updates.append("output_data = ?")
            params.append(output_str)
            
        params.append(task_id)
        query = f"UPDATE agent_tasks SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, tuple(params))
    else:
        now = started_at or datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO agent_tasks (id, project_id, agent_name, status, started_at, completed_at, output_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, project_id, agent_name, status, now, completed_at, output_str)
        )
        
    conn.commit()
    conn.close()

def get_agent_tasks_records(project_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agent_tasks WHERE project_id = ?", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        d["output_data"] = json.loads(d["output_data"]) if d["output_data"] else None
        results.append(d)
    return results

# Logging retrieval helper
def get_execution_logs_records(project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM execution_logs WHERE project_id = ? ORDER BY timestamp ASC LIMIT ?",
        (project_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# App Settings Helpers
def get_setting(key: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT setting_value FROM settings WHERE setting_key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return ""

def set_setting(key: str, value: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO settings (setting_key, setting_value) VALUES (?, ?) ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value",
        (key, value)
    )
    conn.commit()
    conn.close()
    # Also update in-memory settings if they correspond
    if key == "gemini_api_key":
        settings.GEMINI_API_KEY = value
    elif key == "default_model":
        settings.DEFAULT_MODEL = value
