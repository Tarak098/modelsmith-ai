import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record, get_task_output_data
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

logger = logging.getLogger(__name__)

class HyperparameterTunerAgent:
    def __init__(self):
        self.agent_name = "hyperparameter_tuner"

    def execute(self, project_id: str, engineered_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any], best_model_name: str, best_model: Any, model_select_summary: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        log_agent_action(project_id, self.agent_name, "INFO", f"Tuning hyperparameters for model: '{best_model_name}'.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(engineered_file_path)
            target_col = eda_report["target_column"]
            category = plan.get("category", "classification")
            
            # Split X and y
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            from sklearn.preprocessing import RobustScaler
            best_attempt = model_select_summary.get("best_attempt", 1)
            trained_features = model_select_summary.get("best_trained_features", [])
            
            if best_attempt == 2:
                num_cols = list(X.select_dtypes(include=[np.number]).columns)
                num_cols = [c for c in num_cols if c in trained_features]
                if num_cols:
                    scaler = RobustScaler()
                    X_train_processed = X_train.copy()
                    X_val_processed = X_val.copy()
                    X_train_processed[num_cols] = scaler.fit_transform(X_train[num_cols])
                    X_val_processed[num_cols] = scaler.transform(X_val[num_cols])
                    X_train, X_val = X_train_processed, X_val_processed
            elif best_attempt == 3:
                num_cols = list(X.select_dtypes(include=[np.number]).columns)
                num_cols = [c for c in num_cols if c in trained_features]
                X_train_processed = X_train.copy()
                X_val_processed = X_val.copy()
                for col in num_cols:
                    skew = X_train[col].skew()
                    if abs(skew) > 1.0 and X_train[col].min() >= 0:
                        X_train_processed[col] = np.log1p(X_train[col])
                        X_val_processed[col] = np.log1p(X_val[col].clip(lower=0))
                X_train, X_val = X_train_processed, X_val_processed
                
            if trained_features:
                # Deduplicate features and select safely
                valid_features = [c for c in trained_features if c in X_train.columns]
                X_train = X_train[valid_features]
                X_val = X_val[valid_features]
            
            # 1. Fetch Strategy configuration
            strategy = get_task_output_data(project_id, "automl_strategy")
            if not strategy:
                strategy = {
                    "tuning_strategy": "RandomizedSearchCV",
                    "tuning_budget_iters": 10,
                    "cv_folds": 3
                }
                
            tuning_strategy = strategy.get("tuning_strategy", "RandomizedSearchCV")
            tuning_iters = strategy.get("tuning_budget_iters", 10)
            cv_folds = strategy.get("cv_folds", 3)
            
            # If tuning strategy is set to Skip (Dataset > 100k rows)
            if tuning_strategy == "Skip" or tuning_iters == 0:
                log_agent_action(project_id, self.agent_name, "INFO", "Dataset rows > 100,000. Skipping tuning to use default pre-optimized hyperparameters.")
                best_model.fit(X_train, y_train)
                StorageManager.save_model(project_id, best_model, "best_model")
                
                tuned_summary = {
                    "tuned_parameters": best_model.get_params(),
                    "tuning_metrics": {
                        "before_tune": model_select_summary["best_score"],
                        "after_tune": model_select_summary["best_score"]
                    },
                    "improvement_ratio": 0.0,
                    "reasoning": "Tuning skipped due to training time constraints on dataset size exceeding 100,000 rows."
                }
                save_agent_task_record(project_id, self.agent_name, "completed", output_data=tuned_summary)
                return best_model, tuned_summary
                
            # Define hyperparameter distributions for RandomizedSearchCV
            param_dist = {}
            best_model_name_lower = best_model_name.lower()
            
            if "logistic" in best_model_name_lower:
                param_dist = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
            elif "random forest" in best_model_name_lower:
                param_dist = {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10]
                }
            elif "decision tree" in best_model_name_lower:
                param_dist = {
                    "max_depth": [None, 3, 5, 10],
                    "min_samples_split": [2, 5, 10]
                }
            elif "gradient boosting" in best_model_name_lower:
                param_dist = {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5]
                }
            elif "ridge" in best_model_name_lower:
                param_dist = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            elif "lasso" in best_model_name_lower or "elasticnet" in best_model_name_lower:
                param_dist = {"alpha": [0.01, 0.1, 1.0, 10.0]}
            elif "svm" in best_model_name_lower or "svr" in best_model_name_lower:
                param_dist = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
            else:
                param_dist = {}
                
            if not param_dist:
                log_agent_action(project_id, self.agent_name, "WARNING", f"No tuning distributions defined for '{best_model_name}'. Skipping parameter search.")
                tuned_summary = {
                    "tuned_parameters": {},
                    "tuning_metrics": {
                        "before_tune": model_select_summary["best_score"],
                        "after_tune": model_select_summary["best_score"]
                    },
                    "improvement_ratio": 0.0
                }
                save_agent_task_record(project_id, self.agent_name, "completed", output_data=tuned_summary)
                return best_model, tuned_summary
                
            log_agent_action(project_id, self.agent_name, "INFO", f"Searching parameter distributions (n_iter={tuning_iters}): {param_dist}")
            scoring = "f1_macro" if category == "classification" else "r2"
            
            # Run RandomizedSearchCV with early stopping / low iterations bounds
            search = RandomizedSearchCV(
                best_model, 
                param_distributions=param_dist, 
                n_iter=tuning_iters, 
                cv=cv_folds, 
                scoring=scoring, 
                random_state=42,
                n_jobs=1
            )
            search.fit(X_train, y_train)
            
            tuned_model = search.best_estimator_
            best_params = search.best_params_
            
            # Compare performance
            preds_before = best_model.predict(X_val)
            preds_after = tuned_model.predict(X_val)
            
            if category == "classification":
                score_before = float(f1_score(y_val, preds_before, average="macro"))
                score_after = float(f1_score(y_val, preds_after, average="macro"))
            else:
                score_before = float(r2_score(y_val, preds_before))
                score_after = float(r2_score(y_val, preds_after))
                
            diff = score_after - score_before
            improvement_ratio = float(diff / score_before) if score_before != 0 else 0.0
            
            log_agent_action(
                project_id, 
                self.agent_name, 
                "INFO", 
                f"Tuning finished. Best params: {best_params}. Score changed from {score_before:.3f} -> {score_after:.3f} (Change: {diff:+.3f})."
            )
            
            # Save winning tuned estimator
            StorageManager.save_model(project_id, tuned_model, "best_model")
            
            tuned_summary = {
                "tuned_parameters": best_params,
                "tuning_metrics": {
                    "before_tune": score_before,
                    "after_tune": score_after
                },
                "improvement_ratio": improvement_ratio
            }
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=tuned_summary)
            return tuned_model, tuned_summary
            
        except Exception as e:
            error_msg = f"Hyperparameter Tuning failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)
