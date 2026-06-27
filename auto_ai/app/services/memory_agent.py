import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from auto_ai.app.infra.db import get_db_connection
from auto_ai.app.utils.logging import log_agent_action

class MemoryAgent:
    def __init__(self):
        self.agent_name = "memory_agent"

    def record_project_memory(self, project_id: str, project_name: str, category: str, summary_data: Dict[str, Any]):
        """Save a summary of the project run to long-term memory."""
        log_agent_action(project_id, self.agent_name, "INFO", f"Recording project '{project_name}' into long-term memory.")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Save structural context
        memory_key = f"project:{project_id}"
        memory_value = {
            "project_id": project_id,
            "name": project_name,
            "category": category,
            "summary": summary_data,
            "timestamp": datetime.now().isoformat()
        }
        
        cursor.execute(
            "INSERT OR REPLACE INTO memory_store (id, memory_key, memory_value, created_at) VALUES (?, ?, ?, ?)",
            (project_id, memory_key, json.dumps(memory_value), datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
        log_agent_action(project_id, self.agent_name, "INFO", "Project successfully memorized.")

    def search_similar_project(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search memory records for a project that matches the user's natural language request.
        Uses simple keyword overlap matching for semantic search.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT memory_value FROM memory_store")
        rows = cursor.fetchall()
        conn.close()
        
        query_words = set(query.lower().split())
        best_match = None
        highest_overlap = 0
        
        for row in rows:
            data = json.loads(row[0])
            name_desc = (data["name"] + " " + data["summary"].get("description", "")).lower()
            target_words = set(name_desc.split())
            
            overlap = len(query_words.intersection(target_words))
            if overlap > highest_overlap and overlap > 0:
                highest_overlap = overlap
                best_match = data
                
        return best_match

    def list_all_memory_entries(self) -> List[Dict[str, Any]]:
        """List all items stored in the long-term memory store."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_store ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
