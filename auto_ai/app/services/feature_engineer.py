import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record, get_task_output_data
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException
from auto_ai.app.services.preprocessing import PreprocessingPipeline

class FeatureEngineerAgent:
    def __init__(self):
        self.agent_name = "feature_engineer"

    def execute(self, project_id: str, cleaned_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any]) -> Path:
        log_agent_action(project_id, self.agent_name, "INFO", "Executing strategy-driven Feature Engineering pipeline.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(cleaned_file_path)
            target_col = eda_report["target_column"]
            category = plan.get("category", "classification")
            
            # Fetch strategy decided by AutoML Strategy Agent
            strategy = get_task_output_data(project_id, "automl_strategy")
            if not strategy:
                strategy = {
                    "scaler": "StandardScaler",
                    "categorical_encoder": "OneHotEncoder",
                    "cv_folds": 3
                }
                
            transformations = []
            
            # 0. Drop noise features identified by intelligence
            intel_report = get_task_output_data(project_id, "dataset_intelligence")
            cols_to_drop = []
            if intel_report:
                cols_to_drop.extend(intel_report.get("constant_cols", []))
                # Identify ID columns left in dataset
                for col in df.columns:
                    if col == target_col:
                        continue
                    col_lower = col.lower()
                    if any(pat in col_lower for pat in ["passengerid", "uuid", "id", "key", "index", "pk"]):
                        if df[col].nunique() / len(df) > 0.9:
                            cols_to_drop.append(col)
                            
            if cols_to_drop:
                # Deduplicate and drop safely
                cols_to_drop = list(set([c for c in cols_to_drop if c in df.columns]))
                df = df.drop(columns=cols_to_drop)
                log_agent_action(project_id, self.agent_name, "INFO", f"Dropped noise features: {cols_to_drop}")
                transformations.append(f"Dropped noise features: {cols_to_drop}")
                
            # Drop low-correlation features if the dataset has many features (e.g. > 20 columns)
            features = [c for c in df.columns if c != target_col]
            if len(features) > 20:
                log_agent_action(project_id, self.agent_name, "INFO", f"Dataset has a large number of features ({len(features)}). Performing correlation-based feature selection...")
                
                # Temporarily encode target and features to compute correlation matrix
                temp_df = df.copy()
                for col in temp_df.columns:
                    if temp_df[col].dtype == object or temp_df[col].dtype.name == 'category':
                        try:
                            temp_df[col] = LabelEncoder().fit_transform(temp_df[col].astype(str))
                        except:
                            temp_df = temp_df.drop(columns=[col])
                
                if target_col in temp_df.columns:
                    correlations = temp_df.corr()[target_col].abs().drop(target_col, errors='ignore')
                    # Drop columns where correlation is extremely low (< 0.05)
                    low_corr_cols = list(correlations[correlations < 0.05].index)
                    
                    # Ensure we don't drop too many features, leaving at least 15 features
                    max_to_drop = len(features) - 15
                    if len(low_corr_cols) > max_to_drop:
                        # Keep top correlated features
                        sorted_low_corr = correlations.loc[low_corr_cols].sort_values()
                        low_corr_cols = list(sorted_low_corr.index[:max_to_drop])
                        
                    if low_corr_cols:
                        df = df.drop(columns=low_corr_cols)
                        log_agent_action(project_id, self.agent_name, "INFO", f"Dropped {len(low_corr_cols)} low-correlation features (<0.05 correlation with target): {low_corr_cols}")
                        transformations.append(f"Dropped {len(low_corr_cols)} low-correlation features")
                
            # 1. Datetime / Lag Feature Extraction
            date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                log_agent_action(project_id, self.agent_name, "INFO", f"Extracting datetime features from '{date_col}'")
                
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                # Sort values to keep chronological ordering
                df = df.sort_values(by=date_col).reset_index(drop=True)
                
                df['year'] = df[date_col].dt.year
                df['month'] = df[date_col].dt.month
                df['day'] = df[date_col].dt.day
                df['dayofweek'] = df[date_col].dt.dayofweek
                df['quarter'] = df[date_col].dt.quarter
                transformations.append(f"Extracted year, month, day, dayofweek, quarter from '{date_col}'")
                
                # Forecasting lag features
                if category == "forecasting" and target_col in df.columns:
                    for lag in [1, 2, 7]:
                        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
                        transformations.append(f"Generated target lag '{target_col}_lag_{lag}'")
                    df = df.bfill()
                    
                df = df.drop(columns=[date_col])
                
            # 2. Target Variable Categorical Encoding (for Classification tasks)
            is_classification = "classification" in str(intel_report.get("inferred_task", category)).lower()
            if is_classification and df[target_col].dtype == 'object':
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col].astype(str))
                StorageManager.save_model(project_id, le, "target_encoder")
                transformations.append(f"Label encoded classification target '{target_col}'")
                
            # 3. Features Preprocessing Pipeline
            pipeline = PreprocessingPipeline(strategy)
            df = pipeline.fit_transform(df, target_col)
            
            # Save preprocessing pipeline for inference/serving
            StorageManager.save_model(project_id, pipeline, "preprocessing_pipeline")
            transformations.append(f"Fit preprocessing pipeline: scaler={pipeline.scaler_type}, encoder={pipeline.encoder_type}")
            
            # Save engineered dataset
            eng_path = StorageManager.save_dataset(project_id, df, "engineered_data.csv")
            
            summary = {
                "transformations": transformations,
                "engineered_columns": list(df.columns),
                "num_features_scaled": len(pipeline.numeric_cols),
                "num_categories_encoded": len(pipeline.categorical_cols)
            }
            
            log_agent_action(project_id, self.agent_name, "INFO", f"Feature engineering complete. Applied {len(transformations)} transformations.")
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=summary)
            return eng_path
            
        except Exception as e:
            error_msg = f"Feature Engineering failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)
