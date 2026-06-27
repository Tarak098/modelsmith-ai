import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class DataValidatorAgent:
    def __init__(self):
        self.agent_name = "data_validator"

    def execute(self, project_id: str, file_path: Path) -> Dict[str, Any]:
        """
        Validate schema, missing values, duplicates, and target classes.
        """
        log_agent_action(project_id, self.agent_name, "INFO", f"Validating dataset file at {file_path}.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            rows, cols = df.shape
            missing_stats = df.isnull().sum().to_dict()
            missing_pct = {k: float(v / rows) for k, v in missing_stats.items()}
            
            duplicates = int(df.duplicated().sum())
            
            # DataType mapping
            dtypes = {col: str(df[col].dtype) for col in df.columns}
            
            # Numeric/Categorical splits
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
            
            # Report generation
            report = {
                "num_rows": rows,
                "num_cols": cols,
                "missing_counts": missing_stats,
                "missing_percentages": missing_pct,
                "duplicate_count": duplicates,
                "data_types": dtypes,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "data_quality_score": self._compute_quality_score(rows, missing_pct, duplicates)
            }
            
            log_agent_action(
                project_id, 
                self.agent_name, 
                "INFO", 
                f"Validation complete. Found {duplicates} duplicates, {sum(missing_stats.values())} missing values. Quality score: {report['data_quality_score']}%."
            )
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=report)
            return report
            
        except Exception as e:
            error_msg = f"Data Validation failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _compute_quality_score(self, total_rows: int, missing_pct: Dict[str, float], duplicate_count: int) -> int:
        """Calculate a generic data quality score out of 100."""
        score = 100
        
        # Deduct for duplicates (cap at 15 points)
        dup_penalty = min(15, int((duplicate_count / total_rows) * 100)) if total_rows > 0 else 0
        score -= dup_penalty
        
        # Deduct for missing values
        max_missing_pct = max(missing_pct.values()) if missing_pct else 0
        missing_penalty = int(max_missing_pct * 30) # cap at 30 points
        score -= min(30, missing_penalty)
        
        return max(10, score)
