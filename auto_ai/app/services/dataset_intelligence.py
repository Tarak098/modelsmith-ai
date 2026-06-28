import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from auto_ai.app.infra.db import save_agent_task_record, get_setting_bool
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

logger = logging.getLogger(__name__)

class TaskConflictException(Exception):
    def __init__(self, message: str, detected_type: str):
        super().__init__(message)
        self.detected_type = detected_type

class DatasetIntelligenceAgent:
    def __init__(self):
        self.agent_name = "dataset_intelligence"

    def execute(self, project_id: str, plan: Dict[str, Any], dataset_path: str) -> Dict[str, Any]:
        log_agent_action(project_id, self.agent_name, "INFO", "Analyzing dataset characteristics and profiling schema.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(dataset_path)
            
            # Calculate MD5 hash
            import hashlib
            hasher = hashlib.md5()
            with open(dataset_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            dataset_hash = hasher.hexdigest()
            
            # 1. Dataset Profile calculations
            rows, cols = df.shape
            mem_usage_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
            missing_cells = int(df.isnull().sum().sum())
            total_cells = rows * cols
            missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
            duplicate_rows = int(df.duplicated().sum())
            
            # Map column types
            num_cols = []
            cat_cols = []
            date_cols = []
            constant_cols = []
            high_card_cols = []
            
            for col in df.columns:
                col_lower = col.lower()
                nunique = df[col].nunique()
                
                # Check for constant column
                if nunique <= 1:
                    constant_cols.append(col)
                    continue
                    
                # High cardinality categorical checks
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    cat_cols.append(col)
                    unique_pct = nunique / rows
                    if unique_pct > 0.6:
                        high_card_cols.append(col)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Check if date representation
                    if any(pat in col_lower for pat in ["date", "time", "timestamp", "year"]):
                        date_cols.append(col)
                    else:
                        num_cols.append(col)
                else:
                    cat_cols.append(col)
            
            # Outlier detection (IQR method on numeric columns)
            outlier_pct = 0.0
            if num_cols:
                outlier_counts = []
                for col in num_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = df[(df[col] < lower) | (df[col] > upper)]
                    outlier_counts.append(len(outliers))
                outlier_pct = (sum(outlier_counts) / (rows * len(num_cols))) * 100
                
            # 2. Target Column Identification
            target_col = self._identify_target(df, plan)
            
            # Target properties and task type detection
            target_series = df[target_col].dropna()
            target_nunique = target_series.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(target_series)
            
            inferred_task = "regression"
            if target_nunique == 2:
                inferred_task = "binary_classification"
            elif 2 < target_nunique <= 15:
                inferred_task = "multiclass_classification"
            elif is_numeric:
                inferred_task = "regression"
            else:
                inferred_task = "multiclass_classification"
                
            # Check target distribution
            target_distribution = {}
            if inferred_task != "regression" and target_nunique <= 20:
                dist = target_series.value_counts(normalize=True).to_dict()
                target_distribution = {str(k): float(v) for k, v in dist.items()}
                
            # 3. Stronger Data Leakage Detection
            leakage_warnings = []
            if get_setting_bool("enable_data_leakage_detection", True):
                for col in df.columns:
                    if col == target_col:
                        continue
                        
                    col_lower = col.lower()
                    
                    # Target duplicate columns
                    if df[col].equals(df[target_col]):
                        leakage_warnings.append(f"Leakage Warning: Column '{col}' is a duplicate of the target variable '{target_col}'.")
                        
                    # Near perfect correlation leakage
                    if is_numeric and pd.api.types.is_numeric_dtype(df[col]):
                        corr = df[col].corr(df[target_col])
                        if abs(corr) > 0.98:
                            leakage_warnings.append(f"Leakage Warning: Column '{col}' has near-perfect correlation ({corr:.3f}) with target. Might be leakage.")
                            
                    # ID columns left in dataset
                    if any(pat in col_lower for pat in ["passengerid", "uuid", "id", "key", "index", "pk"]):
                        if df[col].nunique() / rows > 0.9:
                            leakage_warnings.append(f"Leakage Warning: Identifier column '{col}' detected. IDs can contaminate validation splits.")
                            
                    # Future timestamp leakage
                    if col in date_cols and any(pat in col_lower for pat in ["end", "expire", "closed", "finish"]):
                        leakage_warnings.append(f"Leakage Warning: Future context timestamp column '{col}' detected.")
            
            # Log leakage warnings
            for w in leakage_warnings:
                log_agent_action(project_id, self.agent_name, "WARNING", w)
                
            # 4. Prompt vs Target Mismatch Task Conflict Check
            # Check prompt keywords
            desc = plan.get("description", "").lower()
            category = plan.get("category", "classification").lower()
            
            is_classification_prompt = any(w in desc for w in ["classify", "classification", "fraud", "churn", "default", "survive", "yes or no"])
            is_regression_prompt = any(w in desc for w in ["predict price", "value", "salary", "amount", "forecast", "regression"])
            
            conflict_detected = False
            conflict_message = ""
            
            if category == "classification" and inferred_task == "regression" and is_regression_prompt:
                conflict_detected = True
                conflict_message = f"ML task type mismatch: Prompt suggests regression/value prediction, but classification was selected. Target '{target_col}' has continuous numeric distribution."
            elif category == "regression" and "classification" in inferred_task and is_classification_prompt:
                conflict_detected = True
                conflict_message = f"ML task type mismatch: Prompt suggests category classification, but regression was selected. Target '{target_col}' has binary/discrete categories."
                
            if conflict_detected:
                log_agent_action(project_id, self.agent_name, "ERROR", f"Task Mismatch: {conflict_message}")
                save_agent_task_record(project_id, self.agent_name, "failed", output_data={
                    "error": conflict_message,
                    "target_col": target_col,
                    "inferred_task": inferred_task
                })
                raise TaskConflictException(conflict_message, inferred_task)
                
            # Formulate full profile report
            profile_report = {
                "dataset_hash": dataset_hash,
                "rows": rows,
                "cols": cols,
                "memory_usage_mb": mem_usage_mb,
                "missing_cells": missing_cells,
                "missing_pct": missing_pct,
                "duplicate_rows": duplicate_rows,
                "numeric_cols": num_cols,
                "categorical_cols": cat_cols,
                "date_cols": date_cols,
                "constant_cols": constant_cols,
                "high_card_cols": high_card_cols,
                "outlier_pct": outlier_pct,
                "target_col": target_col,
                "inferred_task": inferred_task,
                "target_distribution": target_distribution,
                "leakage_warnings": leakage_warnings
            }
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=profile_report)
            log_agent_action(project_id, self.agent_name, "INFO", f"Dataset intelligence profiling complete. target={target_col}, inferred_task={inferred_task}")
            return profile_report
            
        except TaskConflictException as te:
            # Propagate conflict to coordinator
            raise te
        except Exception as e:
            error_msg = f"Dataset Intelligence failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _identify_target(self, df: pd.DataFrame, plan: Dict[str, Any]) -> str:
        cols = list(df.columns)
        desc = plan.get("description", "").lower()
        
        # Check standard targets
        targets = ["outcome", "target", "price", "churn", "attrition", "label", "class", "close", "survived", "defaultrisk", "quality", "medhouseval"]
        for t in targets:
            for col in cols:
                if col.lower() == t:
                    return col
                    
        # Check matching columns in description keywords
        for col in cols:
            if col.lower() in desc:
                return col
                
        # Default to last column
        return cols[-1]
