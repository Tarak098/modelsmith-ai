import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class FeatureEngineerAgent:
    def __init__(self):
        self.agent_name = "feature_engineer"

    def execute(self, project_id: str, cleaned_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any]) -> Path:
        """
        Execute feature transformations (encoding, scaling, datetime extraction) and save.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Executing Feature Engineering steps.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(cleaned_file_path)
            target_col = eda_report["target_column"]
            category = plan.get("category", "classification")
            
            transformations = []
            scalers = {}
            encoders = {}
            
            # 0. Drop identifier/high-cardinality noise columns (which ruin training generalization/accuracy)
            cols_to_drop = []
            for col in df.columns:
                if col == target_col:
                    continue
                col_lower = col.lower()
                
                # Never drop date/time columns needed for forecasting
                if "date" in col_lower or "time" in col_lower:
                    continue
                
                # Check for obvious unique identifier columns (e.g. PassengerId, UUID, Index)
                is_id_name = any(pat in col_lower for pat in ["id", "uuid", "key", "index", "pk"])
                unique_pct = df[col].nunique() / len(df)
                
                # Check for high-cardinality text columns (like names, descriptions, urls)
                is_high_card_text = (df[col].dtype.name in ['object', 'category', 'string', 'str']) and unique_pct > 0.6
                
                # Check for constant/zero-variance features
                is_constant = df[col].nunique() <= 1
                
                if (is_id_name and df[col].nunique() == len(df)) or is_high_card_text or is_constant:
                    cols_to_drop.append(col)
                    
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                log_agent_action(project_id, self.agent_name, "INFO", f"Dropped noise features that degrade generalization/accuracy: {cols_to_drop}")
                transformations.append(f"Dropped noise features: {cols_to_drop}")
            
            # 1. Datetime / Lag Feature Extraction for Forecasting
            date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            if date_cols and category == "forecasting":
                date_col = date_cols[0]
                log_agent_action(project_id, self.agent_name, "INFO", f"Datetime column '{date_col}' detected. Extracting lag and calendar features.")
                
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=date_col).reset_index(drop=True)
                
                # Extract parts
                df['year'] = df[date_col].dt.year
                df['month'] = df[date_col].dt.month
                df['day'] = df[date_col].dt.day
                df['dayofweek'] = df[date_col].dt.dayofweek
                transformations.append(f"Extracted year, month, day, and dayofweek from '{date_col}'")
                
                # Create lag features for numeric columns (excluding datetime and target)
                numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                if target_col in numeric_cols:
                    numeric_cols.remove(target_col)
                    
                # Create lags on target itself if regression/forecasting
                if target_col in df.columns:
                    for lag in [1, 2, 7]:
                        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
                        transformations.append(f"Generated target lag feature '{target_col}_lag_{lag}'")
                    # Fill NaNs created by lagging
                    df = df.bfill()
                
                # Drop original datetime column before training models
                df = df.drop(columns=[date_col])
            
            # 2. Categorical Label Encoding
            cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)
            if target_col in cat_cols:
                # Target encoding (only if it's the target column in classification)
                cat_cols.remove(target_col)
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col].astype(str))
                StorageManager.save_model(project_id, le, "target_encoder")
                transformations.append(f"Label encoded target column '{target_col}'")
                
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                StorageManager.save_model(project_id, le, f"encoder_{col}")
                transformations.append(f"Label encoded feature column '{col}'")
                
            # 3. Numeric scaling (excluding target)
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            if target_col in num_cols:
                num_cols.remove(target_col)
                
            if num_cols:
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[num_cols])
                df[num_cols] = scaled_values
                StorageManager.save_model(project_id, scaler, "features_scaler")
                transformations.append(f"StandardScaled numeric features: {num_cols}")
                
            # Save engineered dataset
            eng_path = StorageManager.save_dataset(project_id, df, "engineered_data.csv")
            
            summary = {
                "transformations": transformations,
                "engineered_columns": list(df.columns),
                "num_features_scaled": len(num_cols),
                "num_categories_encoded": len(cat_cols)
            }
            
            log_agent_action(project_id, self.agent_name, "INFO", f"Feature engineering complete. Applied {len(transformations)} transformations.")
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=summary)
            return eng_path
            
        except Exception as e:
            error_msg = f"Feature Engineering failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)
