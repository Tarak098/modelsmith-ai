import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class HyperparameterTunerAgent:
    def __init__(self):
        self.agent_name = "hyperparameter_tuner"

    def execute(self, project_id: str, engineered_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any], best_model_name: str, best_model: Any, model_select_summary: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning on the selected best model.
        """
        log_agent_action(project_id, self.agent_name, "INFO", f"Tuning hyperparameters for model: '{best_model_name}'.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(engineered_file_path)
            target_col = eda_report["target_column"]
            category = plan.get("category", "classification")
            
            # Split X and y
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Train-Test Split (80% train, 20% validation)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Establish grids depending on the model name
            param_grid = {}
            if "logistic" in best_model_name.lower():
                param_grid = {"C": [0.1, 1.0, 10.0]}
            elif "random forest" in best_model_name.lower():
                param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5, 8]}
            elif "decision tree" in best_model_name.lower():
                param_grid = {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}
            elif "gradient boosting" in best_model_name.lower():
                param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
            elif "ridge" in best_model_name.lower():
                param_grid = {"alpha": [0.1, 1.0, 10.0]}
            else:
                # Default empty grid for other models
                param_grid = {}
                
            if not param_grid:
                log_agent_action(project_id, self.agent_name, "WARNING", f"No tuning grid defined for '{best_model_name}'. Skipping parameter search.")
                tuned_summary = {
                    "tuned_parameters": {},
                    "tuning_metrics": {"before_tune": model_select_summary["best_score"], "after_tune": model_select_summary["best_score"]},
                    "improvement_ratio": 0.0
                }
                save_agent_task_record(project_id, self.agent_name, "completed", output_data=tuned_summary)
                return best_model, tuned_summary
                
            # Perform grid search
            log_agent_action(project_id, self.agent_name, "INFO", f"Searching parameter grid: {param_grid}")
            scoring = "f1_macro" if category == "classification" else "r2"
            
            # Use CV=3 for speed and robustness
            grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring=scoring, n_jobs=1)
            grid_search.fit(X_train, y_train)
            
            tuned_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Compare performance before/after on validation set
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
            
            # Save the optimized model
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
