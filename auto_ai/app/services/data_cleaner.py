import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class DataCleanerAgent:
    def __init__(self):
        self.agent_name = "data_cleaner"

    def execute(self, project_id: str, raw_file_path: Path, validation_report: Dict[str, Any]) -> Path:
        """
        Deduplicates, imputes NaNs, standardizes column names, and saves cleaned dataset.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Starting data cleaning steps.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            # Load dataset
            df = pd.read_csv(raw_file_path)
            initial_rows = len(df)
            
            # 1. Remove duplicates
            df = df.drop_duplicates()
            dedup_count = initial_rows - len(df)
            if dedup_count > 0:
                log_agent_action(project_id, self.agent_name, "INFO", f"Removed {dedup_count} duplicate rows.")
            
            # 2. Clean/Standardize column names
            # Lowercase, replace whitespace/special characters with underscore
            original_cols = list(df.columns)
            new_cols = []
            for col in original_cols:
                clean_col = col.strip().replace(" ", "_").replace("-", "_")
                clean_col = "".join(c for c in clean_col if c.isalnum() or c == "_")
                new_cols.append(clean_col)
            
            df.columns = new_cols
            
            # Map clean column names to validation info
            col_map = dict(zip(original_cols, new_cols))
            numeric_cols = [col_map[c] for c in validation_report["numeric_columns"] if c in col_map]
            categorical_cols = [col_map[c] for c in validation_report["categorical_columns"] if c in col_map]
            
            # 3. Handle missing values (Imputation)
            imputation_summary = {}
            for col in df.columns:
                null_count = int(df[col].isnull().sum())
                if null_count > 0:
                    if col in numeric_cols:
                        # Impute with median
                        median_val = df[col].median()
                        if pd.isnull(median_val):  # All elements were NaN
                            median_val = 0.0
                        df[col] = df[col].fillna(median_val)
                        imputation_summary[col] = f"Imputed {null_count} missing values with median ({median_val})"
                    else:
                        # Impute with mode
                        if not df[col].mode().empty:
                            mode_val = df[col].mode()[0]
                        else:
                            mode_val = "Unknown"
                        df[col] = df[col].fillna(mode_val)
                        imputation_summary[col] = f"Imputed {null_count} missing values with mode ('{mode_val}')"
            
            if imputation_summary:
                log_agent_action(project_id, self.agent_name, "INFO", f"Completed imputation on columns: {list(imputation_summary.keys())}.")
            
            # Save cleaned dataset
            cleaned_path = StorageManager.save_dataset(project_id, df, "cleaned_data.csv")
            
            summary = {
                "initial_rows": initial_rows,
                "cleaned_rows": len(df),
                "removed_duplicates": dedup_count,
                "column_renames": {k: v for k, v in col_map.items() if k != v},
                "imputations": imputation_summary
            }
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=summary)
            return cleaned_path
            
        except Exception as e:
            error_msg = f"Data Cleaning failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)
