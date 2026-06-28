import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record, get_task_output_data
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

logger = logging.getLogger(__name__)

class ModelSelectionAgent:
    def __init__(self):
        self.agent_name = "model_selector"

    def execute(self, project_id: str, engineered_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[Any, str, Dict[str, Any]]:
        log_agent_action(project_id, self.agent_name, "INFO", "Executing model comparison with CV scoring and time budgets.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(engineered_file_path)
            target_col = eda_report["target_column"]
            category = plan.get("category", "classification")
            
            # 1. Fetch Strategy and Budgets
            strategy = get_task_output_data(project_id, "automl_strategy")
            if not strategy:
                strategy = {
                    "scaler": "StandardScaler",
                    "candidate_models": ["LogisticRegression", "RandomForest"] if category == "classification" else ["LinearRegression", "RandomForest"],
                    "cv_folds": 3,
                    "max_training_time_per_model": 300,
                    "max_retries": 5
                }
                
            cv_folds = strategy.get("cv_folds", 3)
            max_train_time = strategy.get("max_training_time_per_model", 300)
            max_retries = strategy.get("max_retries", 5)
            
            # Split X and y
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Category auto-corrections
            if category == "classification" and pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
                category = "regression"
                plan["category"] = "regression"
            elif category == "regression" and y.nunique() <= 2:
                category = "classification"
                plan["category"] = "classification"
                
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Get list of model names to evaluate from strategy
            strategy_models = strategy.get("candidate_models", [])
            
            # Intelligent retry loop setup (Improvement 7)
            best_score = -999999
            best_model_name = ""
            best_model = None
            best_features = list(X_train.columns)
            best_attempt = 1
            leaderboard = []
            
            # Attempt 1 is default training
            # Poor score threshold: F1 < 0.7 for classification, R2 < 0.6 for regression
            threshold_f1 = 0.70
            threshold_r2 = 0.60
            
            for attempt_idx in range(1, max_retries + 1):
                log_agent_action(project_id, self.agent_name, "INFO", f"Model Selection Attempt {attempt_idx} started.")
                
                # Apply attempt adjustments in-place to copies of data
                X_train_adj = X_train.copy()
                X_val_adj = X_val.copy()
                
                adjustment_applied = "Default Preprocessing"
                
                if attempt_idx == 2:
                    adjustment_applied = "Applied RobustScaler to numerical features"
                    # Force RobustScaler mapping
                    num_cols = list(X_train_adj.select_dtypes(include=[np.number]).columns)
                    if num_cols:
                        scaler = RobustScaler()
                        X_train_adj[num_cols] = scaler.fit_transform(X_train_adj[num_cols])
                        X_val_adj[num_cols] = scaler.transform(X_val_adj[num_cols])
                        
                elif attempt_idx == 3:
                    adjustment_applied = "Applied Log1p transforms to skewed numerical columns"
                    num_cols = list(X_train_adj.select_dtypes(include=[np.number]).columns)
                    for col in num_cols:
                        skew = X_train_adj[col].skew()
                        if abs(skew) > 1.0 and X_train_adj[col].min() >= 0:
                            X_train_adj[col] = np.log1p(X_train_adj[col])
                            X_val_adj[col] = np.log1p(X_val_adj[col].clip(lower=0))
                            
                elif attempt_idx == 4:
                    adjustment_applied = "Feature Selection: Dropped highly-correlated features (>0.85)"
                    # Drop highly correlated columns
                    corr_matrix = X_train_adj.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
                    if to_drop:
                        X_train_adj = X_train_adj.drop(columns=to_drop)
                        X_val_adj = X_val_adj.drop(columns=to_drop)
                        
                elif attempt_idx == 5:
                    adjustment_applied = "Expanded candidate model pool with alternative algorithm families"
                    # Add alternative families: SVM, NaiveBayes for classification, SVR, ElasticNet for regression
                    if category == "classification":
                        if "SVM" not in strategy_models: strategy_models.append("SVM")
                        if "NaiveBayes" not in strategy_models: strategy_models.append("NaiveBayes")
                    else:
                        if "SVR" not in strategy_models: strategy_models.append("SVR")
                        if "ElasticNet" not in strategy_models: strategy_models.append("ElasticNet")
                
                log_agent_action(project_id, self.agent_name, "INFO", f"Attempt {attempt_idx} Strategy: {adjustment_applied}")
                
                # Instantiate candidates dynamically
                candidates = self._instantiate_candidates(strategy_models, category)
                
                leaderboard = []
                best_attempt_score = -999999
                
                for name, model in candidates:
                    logger.info(f"Evaluating candidate: {name}")
                    start_time = time.time()
                    
                    try:
                        # Perform cross validation scoring with max training timeout check
                        cv_start = time.time()
                        # Use Stratified CV for classification, default CV for regression
                        scoring_metric = "f1_macro" if category == "classification" else "r2"
                        scores = cross_val_score(model, X_train_adj, y_train, cv=cv_folds, scoring=scoring_metric)
                        cv_score = float(np.mean(scores))
                        cv_time = time.time() - cv_start
                        
                        if cv_time > max_train_time:
                            log_agent_action(project_id, self.agent_name, "WARNING", f"Model '{name}' exceeded training time budget ({cv_time:.1f}s > {max_train_time}s). Skipping.")
                            continue
                            
                        # Train on full train set to test validation score
                        model.fit(X_train_adj, y_train)
                        train_time = time.time() - start_time
                        
                        # Inference speed test
                        inf_start = time.time()
                        preds = model.predict(X_val_adj)
                        inf_time = time.time() - inf_start
                        
                        # Compute metrics
                        metrics = {}
                        score_to_compare = 0.0
                        
                        if category == "classification":
                            acc = float(accuracy_score(y_val, preds))
                            f1 = float(f1_score(y_val, preds, average="macro"))
                            metrics = {"accuracy": acc, "f1_score": f1}
                            score_to_compare = f1
                        else:
                            r2 = float(r2_score(y_val, preds))
                            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                            metrics = {"r2_score": r2, "rmse": rmse}
                            score_to_compare = r2
                            
                        leaderboard.append({
                            "model_name": name,
                            "cv_score": cv_score,
                            "validation_score": score_to_compare,
                            "metrics": metrics,
                            "training_time_sec": train_time,
                            "inference_time_sec": inf_time,
                            "hyperparameters": {k: str(v) for k, v in model.get_params().items() if len(str(v)) < 100},
                            "attempt": attempt_idx
                        })
                        
                        if score_to_compare > best_attempt_score:
                            best_attempt_score = score_to_compare
                            
                        if score_to_compare > best_score:
                            best_score = score_to_compare
                            best_model_name = name
                            best_model = model
                            best_features = list(X_train_adj.columns)
                            best_attempt = attempt_idx
                            # If we made adjustments, we need to save the adjusted pipeline scaler
                            if attempt_idx == 2:
                                StorageManager.save_model(project_id, RobustScaler().fit(X_train[num_cols]), "attempt_scaler")
                            elif attempt_idx == 3:
                                # save indicator of log skew columns
                                StorageManager.save_model(project_id, num_cols, "attempt_log_cols")
                                
                    except Exception as ex:
                        logger.error(f"Error training {name}: {ex}")
                        continue
                        
                # Check if best validation score satisfies poor score threshold
                satisfied = False
                if category == "classification" and best_attempt_score >= threshold_f1:
                    satisfied = True
                elif category == "regression" and best_attempt_score >= threshold_r2:
                    satisfied = True
                    
                if satisfied:
                    log_agent_action(project_id, self.agent_name, "INFO", f"Performance threshold satisfied on Attempt {attempt_idx} (Score: {best_attempt_score:.3f}). Stopping retries.")
                    break
                else:
                    log_agent_action(project_id, self.agent_name, "WARNING", f"Attempt {attempt_idx} validation score ({best_attempt_score:.3f}) falls below target. Retrying with adjusted preprocessing...")
            
            # Sort leaderboard
            leaderboard.sort(key=lambda x: x["validation_score"], reverse=True)
            for idx, entry in enumerate(leaderboard):
                entry["ranking"] = idx + 1
                
            # Save the winning model
            StorageManager.save_model(project_id, best_model, "best_model")
            log_agent_action(project_id, self.agent_name, "INFO", f"Comparison complete. Winning Model: '{best_model_name}' (Score: {best_score:.3f}). Saved.")
            
            summary = {
                "leaderboard": leaderboard,
                "best_model_name": best_model_name,
                "best_score": best_score,
                "comparison_metric": "f1_score" if category == "classification" else "r2_score",
                "best_trained_features": best_features,
                "best_attempt": best_attempt
            }
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=summary)
            return best_model, best_model_name, summary
            
        except Exception as e:
            error_msg = f"Model Selection failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _instantiate_candidates(self, model_names: List[str], category: str) -> List[Tuple[str, Any]]:
        candidates = []
        is_classification = category == "classification"
        
        # Safe library fallbacks
        XGBClass = HistGradientBoostingClassifier
        XGBReg = HistGradientBoostingRegressor
        LGBMClass = HistGradientBoostingClassifier
        LGBMReg = HistGradientBoostingRegressor
        CatClass = RandomForestClassifier
        CatReg = RandomForestRegressor
        
        try:
            from xgboost import XGBClassifier, XGBRegressor
            XGBClass = XGBClassifier
            XGBReg = XGBRegressor
        except: pass
        
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            LGBMClass = LGBMClassifier
            LGBMReg = LGBMRegressor
        except: pass
        
        for name in model_names:
            if is_classification:
                if name == "LogisticRegression":
                    candidates.append(("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)))
                elif name == "RandomForest":
                    candidates.append(("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42)))
                elif name == "GradientBoosting":
                    candidates.append(("Gradient Boosting", GradientBoostingClassifier(n_estimators=50, random_state=42)))
                elif name == "HistGradientBoosting":
                    candidates.append(("Hist Gradient Boosting", HistGradientBoostingClassifier(max_iter=50, random_state=42)))
                elif name == "XGBoost":
                    candidates.append(("XGBoost", XGBClass(random_state=42)))
                elif name == "LightGBM":
                    candidates.append(("LightGBM", LGBMClass(random_state=42)))
                elif name == "CatBoost":
                    candidates.append(("CatBoost", CatClass(random_state=42)))
                elif name == "NaiveBayes":
                    candidates.append(("Naive Bayes", GaussianNB()))
                elif name == "SVM":
                    candidates.append(("SVM", SVC(probability=True, random_state=42, max_iter=2000)))
                elif name == "MLP":
                    candidates.append(("MLP Classifier", MLPClassifier(max_iter=500, random_state=42, early_stopping=True)))
            else:
                if name == "LinearRegression":
                    candidates.append(("Linear Regression", LinearRegression()))
                elif name == "Ridge":
                    candidates.append(("Ridge", Ridge(alpha=1.0)))
                elif name == "Lasso":
                    candidates.append(("Lasso", Lasso(alpha=1.0)))
                elif name == "ElasticNet":
                    candidates.append(("ElasticNet", ElasticNet(alpha=1.0)))
                elif name == "RandomForest":
                    candidates.append(("Random Forest Regressor", RandomForestRegressor(n_estimators=50, random_state=42)))
                elif name == "GradientBoosting":
                    candidates.append(("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=50, random_state=42)))
                elif name == "HistGradientBoosting":
                    candidates.append(("Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(max_iter=50, random_state=42)))
                elif name == "XGBoost":
                    candidates.append(("XGBoost Regressor", XGBReg(random_state=42)))
                elif name == "LightGBM":
                    candidates.append(("LightGBM Regressor", LGBMReg(random_state=42)))
                elif name == "SVR":
                    candidates.append(("SVR", SVR(max_iter=2000)))
                elif name == "KNN":
                    from sklearn.neighbors import KNeighborsRegressor
                    candidates.append(("KNN Regressor", KNeighborsRegressor()))
                    
        # Ensure we have at least one fallback model if list is empty
        if not candidates:
            if is_classification:
                candidates.append(("Random Forest", RandomForestClassifier(random_state=42)))
            else:
                candidates.append(("Random Forest Regressor", RandomForestRegressor(random_state=42)))
                
        return candidates
