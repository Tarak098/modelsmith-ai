import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, auc
)
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class EvaluationAgent:
    def __init__(self):
        self.agent_name = "evaluator"

    def execute(self, project_id: str, engineered_file_path: Path, eda_report: Dict[str, Any], plan: Dict[str, Any], best_model: Any) -> Dict[str, Any]:
        """
        Evaluate final tuned model, generate charts, and run diagnoses for overfitting/underfitting.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Executing final model evaluation.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(engineered_file_path)
            target_col = eda_report["target_column"]
            category = plan.get("category", "classification")
            
            # Split X and y
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Split same way
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            from sklearn.preprocessing import RobustScaler
            from auto_ai.app.infra.db import get_task_output_data
            model_select_summary = get_task_output_data(project_id, "model_selector") or {}
            
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
                valid_features = [c for c in trained_features if c in X_train.columns]
                X_train = X_train[valid_features]
                X_val = X_val[valid_features]
            
            # Predict
            train_preds = best_model.predict(X_train)
            val_preds = best_model.predict(X_val)
            
            metrics = {}
            diagnoses = []
            
            if category == "classification":
                # F1, Acc, Precision, Recall
                train_acc = accuracy_score(y_train, train_preds)
                val_acc = accuracy_score(y_val, val_preds)
                
                prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro")
                
                # Check ROC-AUC (only if binary target for simplicity)
                roc_auc = 0.5
                is_binary = len(np.unique(y)) == 2
                if is_binary:
                    try:
                        # try predicting probabilities
                        val_probs = best_model.predict_proba(X_val)[:, 1]
                        roc_auc = float(roc_auc_score(y_val, val_probs))
                        self._save_roc_curve(project_id, y_val, val_probs)
                    except Exception:
                        # fallback if model doesn't support predict_proba
                        roc_auc = float(roc_auc_score(y_val, val_preds))
                
                metrics = {
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1_score": float(f1),
                    "roc_auc": roc_auc
                }
                
                # Diagnostics
                if train_acc - val_acc > 0.15:
                    diagnoses.append("WARNING: Potential Overfitting detected. Train accuracy is significantly higher than validation accuracy.")
                elif val_acc < 0.55:
                    diagnoses.append("WARNING: Underfitting detected. Validation accuracy is low.")
                    
                # Save Confusion Matrix Plot
                self._save_confusion_matrix(project_id, y_val, val_preds)
                
            else:  # regression / forecasting
                train_r2 = r2_score(y_train, train_preds)
                val_r2 = r2_score(y_val, val_preds)
                
                val_mae = mean_absolute_error(y_val, val_preds)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                
                metrics = {
                    "train_r2": float(train_r2),
                    "val_r2": float(val_r2),
                    "mae": float(val_mae),
                    "rmse": float(val_rmse)
                }
                
                # Diagnostics
                if train_r2 - val_r2 > 0.15:
                    diagnoses.append("WARNING: Potential Overfitting detected. Train R² score is significantly higher than validation R².")
                elif val_r2 < 0.40:
                    diagnoses.append("WARNING: Underfitting detected. R² score is low, model is unable to explain variance well.")
                    
                # Save Residuals Plot
                self._save_residuals_plot(project_id, y_val, val_preds)
                
            if not diagnoses:
                diagnoses.append("SUCCESS: Model shows healthy fit. No significant underfitting or overfitting detected.")
                
            eval_report = {
                "metrics": metrics,
                "diagnoses": diagnoses,
                "charts": {
                    "confusion_matrix" if category == "classification" else "residual_plot": f"plots/{'confusion_matrix' if category == 'classification' else 'residual_plot'}.png",
                }
            }
            if category == "classification" and is_binary:
                eval_report["charts"]["roc_curve"] = "plots/roc_curve.png"
                
            log_agent_action(project_id, self.agent_name, "INFO", f"Model evaluation complete. Diagnoses: {diagnoses}")
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=eval_report)
            return eval_report
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _save_confusion_matrix(self, project_id: str, y_true: np.ndarray, y_pred: np.ndarray):
        """Draw and save confusion matrix plot."""
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix heatmap
        im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)
        
        # Set ticks
        classes = np.unique(y_true)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        
        # Add labels in cells
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > (cm.max()/2) else "black")
                
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        dest_path = StorageManager.get_plot_path(project_id, "confusion_matrix")
        plt.savefig(dest_path, dpi=120)
        plt.close()

    def _save_roc_curve(self, project_id: str, y_true: np.ndarray, y_prob: np.ndarray):
        """Draw and save ROC curve plot."""
        plt.figure(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        dest_path = StorageManager.get_plot_path(project_id, "roc_curve")
        plt.savefig(dest_path, dpi=120)
        plt.close()

    def _save_residuals_plot(self, project_id: str, y_true: np.ndarray, y_pred: np.ndarray):
        """Draw and save residual plot for regression validation."""
        plt.figure(figsize=(5, 4))
        residuals = y_true - y_pred
        
        plt.scatter(y_pred, residuals, alpha=0.5, color='#319795', edgecolors='black')
        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Residual Analysis Plot')
        plt.tight_layout()
        
        dest_path = StorageManager.get_plot_path(project_id, "residual_plot")
        plt.savefig(dest_path, dpi=120)
        plt.close()
