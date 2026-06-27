import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class ModelSelectionAgent:
    def __init__(self):
        self.agent_name = "model_selector"

    def execute(self, project_id: str, engineered_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[Any, str, Dict[str, Any]]:
        """
        Train multiple candidate models and select the best-performing one.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Starting training and model comparison.")
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
            
            # Select candidates based on task category
            candidates = []
            if category == "classification":
                candidates = [
                    ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
                    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
                    ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=50)),
                    ("Gradient Boosting", GradientBoostingClassifier(random_state=42, n_estimators=50))
                ]
            else:  # regression / forecasting
                candidates = [
                    ("Linear Regression", LinearRegression()),
                    ("Ridge", Ridge(alpha=1.0)),
                    ("Decision Tree Regressor", DecisionTreeRegressor(random_state=42)),
                    ("Random Forest Regressor", RandomForestRegressor(random_state=42, n_estimators=50)),
                    ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42, n_estimators=50))
                ]
                
            leaderboard = []
            best_score = -999999
            best_model_name = ""
            best_model = None
            
            # Train and evaluate each candidate
            for name, model in candidates:
                log_agent_action(project_id, self.agent_name, "INFO", f"Training candidate model: '{name}'")
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                
                metrics = {}
                score_to_compare = 0.0
                
                if category == "classification":
                    acc = float(accuracy_score(y_val, preds))
                    f1 = float(f1_score(y_val, preds, average="macro"))
                    metrics = {"accuracy": acc, "f1_score": f1}
                    score_to_compare = f1
                    log_agent_action(project_id, self.agent_name, "INFO", f"[{name}] Acc: {acc:.3f}, F1 (macro): {f1:.3f}")
                else:
                    r2 = float(r2_score(y_val, preds))
                    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                    metrics = {"r2_score": r2, "rmse": rmse}
                    score_to_compare = r2
                    log_agent_action(project_id, self.agent_name, "INFO", f"[{name}] R²: {r2:.3f}, RMSE: {rmse:.3f}")
                    
                leaderboard.append({
                    "model_name": name,
                    "metrics": metrics,
                    "comparison_score": score_to_compare
                })
                
                # Check if this is the best model
                if score_to_compare > best_score:
                    best_score = score_to_compare
                    best_model_name = name
                    best_model = model
            
            # Save the best model
            model_key = best_model_name.lower().replace(" ", "_")
            StorageManager.save_model(project_id, best_model, "best_model")
            log_agent_action(project_id, self.agent_name, "INFO", f"Comparison complete. Best Model: '{best_model_name}' (Score: {best_score:.3f}). Saved as 'best_model.joblib'.")
            
            summary = {
                "leaderboard": leaderboard,
                "best_model_name": best_model_name,
                "best_score": best_score,
                "comparison_metric": "f1_score" if category == "classification" else "r2_score"
            }
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=summary)
            return best_model, best_model_name, summary
            
        except Exception as e:
            error_msg = f"Model Selection failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)
