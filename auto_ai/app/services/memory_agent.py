import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from auto_ai.app.infra.db import get_db_connection
from auto_ai.app.utils.logging import log_agent_action

logger = logging.getLogger(__name__)

class MemoryAgent:
    def __init__(self):
        self.agent_name = "memory_agent"

    def record_project_memory(self, project_id: str, project_name: str, category: str, summary_data: Dict[str, Any]):
        log_agent_action(project_id, self.agent_name, "INFO", f"Recording project '{project_name}' into long-term memory.")
        
        # Read from other task outputs to enrich memory context
        from auto_ai.app.infra.db import get_task_output_data
        intel = get_task_output_data(project_id, "dataset_intelligence") or {}
        collector = get_task_output_data(project_id, "data_collector") or {}
        strategy = get_task_output_data(project_id, "automl_strategy") or {}
        engineer = get_task_output_data(project_id, "feature_engineer") or {}
        tuner = get_task_output_data(project_id, "hyperparameter_tuner") or {}
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        memory_key = f"project:{project_id}"
        memory_value = {
            "project_id": project_id,
            "name": project_name,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": intel.get("dataset_hash", ""),
            "dataset_source": collector.get("source", "unknown"),
            "description": summary_data.get("description", ""),
            "best_preprocessing": {
                "scaler": strategy.get("scaler", "StandardScaler"),
                "categorical_encoder": strategy.get("categorical_encoder", "OneHotEncoder")
            },
            "best_feature_engineering": engineer.get("transformations", []),
            "best_model": summary_data.get("model_name", "Unknown"),
            "best_parameters": tuner.get("tuned_parameters", {}),
            "evaluation_metrics": summary_data.get("metrics", {})
        }
        
        cursor.execute(
            "INSERT OR REPLACE INTO memory_store (id, memory_key, memory_value, created_at) VALUES (?, ?, ?, ?)",
            (project_id, memory_key, json.dumps(memory_value), datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
        log_agent_action(project_id, self.agent_name, "INFO", "Project successfully memorized.")

    def search_by_dataset_hash(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        if not dataset_hash:
            return None
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT memory_value FROM memory_store")
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            data = json.loads(row[0])
            if data.get("dataset_hash") == dataset_hash:
                logger.info(f"Memory Hit on dataset hash: {dataset_hash}")
                return data
        return None

    def search_similar_project(self, query: str) -> Optional[Dict[str, Any]]:
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
            name_desc = (data["name"] + " " + data.get("description", "")).lower()
            target_words = set(name_desc.split())
            
            overlap = len(query_words.intersection(target_words))
            if overlap > highest_overlap and overlap > 0:
                highest_overlap = overlap
                best_match = data
                
        return best_match

    def list_all_memory_entries(self) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_store ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
