import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from auto_ai.app.infra.db import get_setting, get_setting_bool, get_setting_float, get_setting_int
from auto_ai.app.services.dataset_cache import DatasetCache
from auto_ai.app.services.dataset_ranker import DatasetRanker
from auto_ai.app.services.openml_adapter import OpenMLAdapter
from auto_ai.app.services.kaggle_adapter import KaggleAdapter
from auto_ai.app.services.uci_adapter import UCIAdapter
from auto_ai.app.services.sklearn_adapter import SklearnAdapter

logger = logging.getLogger(__name__)

class DatasetRetrievalAgent:
    def __init__(self):
        self.adapters = {
            "openml": OpenMLAdapter(),
            "kaggle": KaggleAdapter(),
            "uci": UCIAdapter(),
            "sklearn": SklearnAdapter()
        }

    def execute(self, project_id: str, query: str, expected_task: str = "classification") -> Optional[str]:
        from auto_ai.app.utils.logging import log_agent_action
        
        log_agent_action(project_id, "data_collector", "INFO", f"Starting Dataset Search & Retrieval for query: '{query}'")
        
        cached_file = DatasetCache.check(query)
        if cached_file:
            log_agent_action(project_id, "data_collector", "INFO", f"Local Cache Hit. Found matching cached file: {cached_file}")
            return cached_file
            
        priority_str = get_setting("repository_priority") or "cache,kaggle,openml,uci,sklearn"
        priority_list = [p.strip().lower() for p in priority_str.split(",")]
        threshold = get_setting_float("dataset_score_threshold", 0.4)
        max_candidates = get_setting_int("max_candidate_datasets", 10)
        
        log_agent_action(project_id, "data_collector", "INFO", f"Search configuration loaded: priority={priority_list}, threshold={threshold}")
        
        for repo in priority_list:
            if repo == "cache":
                continue
                
            is_enabled = get_setting_bool(f"enable_{repo}", True)
            if not is_enabled:
                log_agent_action(project_id, "data_collector", "INFO", f"Repository '{repo}' is disabled. Skipping.")
                continue
                
            adapter = self.adapters.get(repo)
            if not adapter:
                continue
                
            log_agent_action(project_id, "data_collector", "INFO", f"Searching remote repository: '{repo}'...")
            try:
                candidates = adapter.search(query, limit=max_candidates)
                if not candidates:
                    log_agent_action(project_id, "data_collector", "INFO", f"No candidate datasets returned from repository '{repo}'")
                    continue
                    
                ranked_candidates = []
                for c in candidates:
                    score = DatasetRanker.score(query, c, expected_task=expected_task)
                    ranked_candidates.append((score, c))
                    
                ranked_candidates.sort(key=lambda x: x[0], reverse=True)
                
                if ranked_candidates:
                    best_score, best_candidate = ranked_candidates[0]
                    log_agent_action(project_id, "data_collector", "INFO", f"Repository '{repo}' top candidate: '{best_candidate['name']}' (Score: {best_score:.3f})")
                    
                    if best_score >= threshold:
                        log_agent_action(project_id, "data_collector", "INFO", f"Candidate '{best_candidate['name']}' meets suitability threshold ({best_score:.3f} >= {threshold}). Downloading dataset...")
                        
                        cache_dir = DatasetCache.get_cache_dir()
                        safe_name = "".join(c for c in best_candidate['name'] if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()
                        if not safe_name.endswith('.csv'):
                            safe_name += '.csv'
                        dest_path = cache_dir / safe_name
                        
                        success = adapter.download(best_candidate["download_url"], str(dest_path))
                        if success and dest_path.exists():
                            log_agent_action(project_id, "data_collector", "INFO", f"Successfully downloaded and cached: {dest_path.name}")
                            DatasetCache.add(best_candidate['name'], str(dest_path))
                            return str(dest_path)
                        else:
                            log_agent_action(project_id, "data_collector", "WARNING", f"Download failed for '{best_candidate['name']}' from '{repo}'. Continuing search...")
                    else:
                        log_agent_action(project_id, "data_collector", "INFO", f"Top candidate score {best_score:.3f} is below threshold {threshold}. Continuing search...")
            except Exception as e:
                log_agent_action(project_id, "data_collector", "ERROR", f"Error during retrieval search from repository '{repo}': {e}")
                
        log_agent_action(project_id, "data_collector", "WARNING", "No suitable public repository dataset found above suitability score threshold.")
        return None
