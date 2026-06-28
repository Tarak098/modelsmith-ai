import logging
from typing import Dict, Any, List
from auto_ai.app.infra.db import save_agent_task_record, get_setting_int, get_setting_bool
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

logger = logging.getLogger(__name__)

class AutoMLStrategyAgent:
    def __init__(self):
        self.agent_name = "automl_strategy"

    def execute(self, project_id: str, profile_report: Dict[str, Any]) -> Dict[str, Any]:
        log_agent_action(project_id, self.agent_name, "INFO", "Formulating AutoML modeling and preprocessing strategy.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            rows = profile_report.get("rows", 0)
            cols = profile_report.get("cols", 0)
            inferred_task = profile_report.get("inferred_task", "classification")
            outlier_pct = profile_report.get("outlier_pct", 0.0)
            high_card_cols = profile_report.get("high_card_cols", [])
            
            strategy = {}
            
            # 1. Decide scaling method (Problem 4 / Improvement 3)
            # If high outlier count, use RobustScaler to neutralize outlier variance
            if outlier_pct > 5.0:
                strategy["scaler"] = "RobustScaler"
                strategy["scaler_reason"] = f"RobustScaler chosen because outliers constitute {outlier_pct:.1f}% of numeric distribution."
            else:
                strategy["scaler"] = "StandardScaler"
                strategy["scaler_reason"] = "StandardScaler chosen for general unit variance normalization of numeric features."
                
            # 2. Decide categorical encoders
            if high_card_cols:
                strategy["categorical_encoder"] = "TargetEncoder"
                strategy["encoder_reason"] = f"TargetEncoder chosen to prevent dimensions explosion from high-cardinality columns: {high_card_cols}."
            else:
                strategy["categorical_encoder"] = "OneHotEncoder"
                strategy["encoder_reason"] = "OneHotEncoder chosen because all categorical columns have low cardinality."
                
            # 3. Choose candidate models (Improvement 4)
            candidates = []
            is_classification = "classification" in inferred_task
            
            if is_classification:
                if rows < 1000:
                    candidates = ["LogisticRegression", "RandomForest", "NaiveBayes"]
                elif rows > 50000:
                    candidates = ["LightGBM", "HistGradientBoosting", "RandomForest"]
                elif len(high_card_cols) > 3 or cols > 50:
                    # High dimensional or sparse
                    candidates = ["LogisticRegression", "SVM", "RandomForest"]
                else:
                    candidates = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost"]
            else: # Regression
                if rows < 1000:
                    # Small dataset
                    candidates = ["LinearRegression", "Ridge", "RandomForest"]
                elif rows > 50000:
                    candidates = ["LightGBM", "HistGradientBoosting", "RandomForest"]
                elif cols > 50:
                    candidates = ["Lasso", "ElasticNet", "RandomForest"]
                else:
                    # Mid-size default
                    candidates = ["LinearRegression", "RandomForest", "XGBoost", "GradientBoosting"]
                    
            strategy["candidate_models"] = candidates
            strategy["model_selection_reason"] = f"Selected {candidates} based on task={inferred_task}, size={rows}x{cols}."
            
            # 4. Hyperparameter tuning budgets (Improvement 5)
            if rows < 20000:
                strategy["tuning_strategy"] = "RandomizedSearchCV"
                strategy["tuning_budget_iters"] = 10
                strategy["tuning_reason"] = f"RandomizedSearchCV selected with 10 iterations (dataset rows={rows} < 20,000)."
            elif 20000 <= rows <= 100000:
                strategy["tuning_strategy"] = "Lightweight"
                strategy["tuning_budget_iters"] = 3
                strategy["tuning_reason"] = f"Lightweight search selected with 3 iterations to enforce runtime speed (dataset rows={rows} between 20k and 100k)."
            else:
                strategy["tuning_strategy"] = "Skip"
                strategy["tuning_budget_iters"] = 0
                strategy["tuning_reason"] = f"Skipping hyperparameter tuning to utilize default pre-optimized hyperparameters (dataset rows={rows} > 100,000)."
                
            # 5. Global Time Budgets (Improvement 6)
            strategy["max_training_time_per_model"] = get_setting_int("max_training_time", 300)
            strategy["max_project_time"] = get_setting_int("max_project_time", 1200)
            strategy["max_retries"] = get_setting_int("max_retries", 5)
            strategy["cv_folds"] = get_setting_int("cv_folds", 3)
            
            # Save task and log strategy
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=strategy)
            log_agent_action(project_id, self.agent_name, "INFO", f"Strategy formulated successfully. scaler={strategy['scaler']}, tuning={strategy['tuning_strategy']}, CV={strategy['cv_folds']} folds")
            return strategy
            
        except Exception as e:
            error_msg = f"AutoML Strategy Formulation failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)
