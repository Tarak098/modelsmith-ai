import logging
import sqlite3
from datetime import datetime
from auto_ai.app.config import settings

# Configure standard console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("modelsmith")

def log_agent_action(project_id: str, agent_name: str, level: str, message: str):
    """
    Log an agent's execution message both to python logging and directly to SQLite database.
    This database storage allows the frontend dashboard to fetch live execution telemetry.
    """
    # Console output
    log_func = logger.info
    if level == "WARNING":
        log_func = logger.warning
    elif level == "ERROR":
        log_func = logger.error
    
    log_func(f"[{project_id}] [{agent_name}] {message}")
    
    # SQLite Output (Done via direct connection to bypass circular references with db modules)
    try:
        conn = sqlite3.connect(str(settings.DB_PATH))
        cursor = conn.cursor()
        
        # Ensure table exists (just in case)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute(
            "INSERT INTO execution_logs (project_id, agent_name, level, message, timestamp) VALUES (?, ?, ?, ?, ?)",
            (project_id, agent_name, level, message, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to write execution log to database: {e}")
