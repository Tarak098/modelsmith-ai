import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class ExplainabilityAgent:
    def __init__(self):
        self.agent_name = "explainability"

    def execute(self, project_id: str, engineered_file_path: Path, eda_report: Dict[str, Any], best_model: Any, best_model_name: str) -> Dict[str, Any]:
        """
        Extract model coefficients or tree feature importances, plot them, and formulate textual explanations.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Executing model explainability analysis.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(engineered_file_path)
            target_col = eda_report["target_column"]
            from auto_ai.app.infra.db import get_task_output_data
            model_select_summary = get_task_output_data(project_id, "model_selector") or {}
            trained_features = model_select_summary.get("best_trained_features", [])
            if trained_features:
                feature_names = [c for c in trained_features if c in df.columns]
            else:
                feature_names = [col for col in df.columns if col != target_col]
            
            importances = []
            
            # 1. Try tree-based importance
            if hasattr(best_model, "feature_importances_"):
                importances = list(best_model.feature_importances_)
            # 2. Try linear coefficients
            elif hasattr(best_model, "coef_"):
                coefs = best_model.coef_
                # coefs could be multi-dimensional for multi-class classification
                if len(coefs.shape) > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)
                else:
                    coefs = np.abs(coefs)
                # Normalize
                sum_coefs = np.sum(coefs)
                importances = list(coefs / sum_coefs) if sum_coefs > 0 else list(coefs)
            else:
                # Fallback: uniform importance if not directly inspectable
                importances = [1.0 / len(feature_names)] * len(feature_names)
                
            # Create mapping
            feature_imp_map = dict(zip(feature_names, importances))
            sorted_imp = sorted(feature_imp_map.items(), key=lambda x: x[1], reverse=True)
            
            # 3. Save Feature Importance Plot
            self._save_feature_importance_plot(project_id, sorted_imp[:10])
            
            # 4. Generate textual limitations and explanations
            limitations = self._get_model_limitations(best_model_name)
            explanation = self._generate_explanation(best_model_name, sorted_imp[:3])
            
            # 5. Extract strategy and intelligence reasoning logs (Improvement 12)
            from auto_ai.app.infra.db import get_task_output_data
            strategy = get_task_output_data(project_id, "automl_strategy") or {}
            intel = get_task_output_data(project_id, "dataset_intelligence") or {}
            category = intel.get("inferred_task", "classification")
            
            reasoning_logs = {
                "target_selection": f"Target column '{target_col}' was automatically selected by analyzing column names against the problem context description.",
                "preprocessing": strategy.get("scaler_reason", "StandardScaler was chosen to normalize numeric distributions."),
                "encoder": strategy.get("encoder_reason", "OneHotEncoder was chosen to process categorical features."),
                "model_selection": strategy.get("model_selection_reason", f"Selected matching model candidate architectures for the '{category}' task."),
                "tuning": strategy.get("tuning_reason", "Hyperparameter tuning budget scaled dynamically to stay within the training time budget.")
            }
            
            report = {
                "feature_importances": {k: float(v) for k, v in sorted_imp},
                "limitations": limitations,
                "natural_language_explanation": explanation,
                "reasoning_logs": reasoning_logs,
                "charts": {
                    "feature_importance": "plots/feature_importance.png"
                }
            }
            
            log_agent_action(project_id, self.agent_name, "INFO", "Explainability analysis finished. Top features plotted.")
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=report)
            return report
            
        except Exception as e:
            error_msg = f"Explainability failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _save_feature_importance_plot(self, project_id: str, top_features: List[tuple]):
        """Draw and save feature importance horizontal bar chart."""
        plt.figure(figsize=(6, 4))
        
        features, values = zip(*reversed(top_features))
        
        plt.barh(features, values, color='#4A5568')
        plt.xlabel('Importance Weight')
        plt.title('Top Predictive Features')
        plt.tight_layout()
        
        dest_path = StorageManager.get_plot_path(project_id, "feature_importance")
        plt.savefig(dest_path, dpi=120)
        plt.close()

    def _get_model_limitations(self, model_name: str) -> List[str]:
        """Provide heuristic limitations of selected ML algorithms."""
        name = model_name.lower()
        if "logistic" in name or "linear" in name or "ridge" in name:
            return [
                "Assumes a linear relationship between features and the target variable.",
                "Highly sensitive to multicollinearity and outlier observations.",
                "Cannot naturally model complex non-linear interaction terms without manual expansions."
            ]
        elif "random forest" in name or "gradient boosting" in name:
            return [
                "Ensemble tree models cannot extrapolate predictions beyond the range of training labels.",
                "Prone to overfitting if tree depth parameter is left entirely unrestricted.",
                "Requires more computation resources for inference compared to linear structures."
            ]
        else:
            return [
                "May struggle to capture highly complex relational patterns.",
                "Verify input distributions regularly for covariate shift."
            ]

    def _generate_explanation(self, model_name: str, top_features: List[tuple]) -> str:
        """Formulate a natural language text explaining model dynamics."""
        f_list = [f"'{feat}' (weight: {val:.2f})" for feat, val in top_features]
        return (
            f"The trained '{model_name}' relies heavily on {', '.join(f_list)} to make decisions. "
            "Increasing values in these top columns strongly influence the final model prediction. "
            "When deploying this model, ensure that data pipelines feeding these particular fields "
            "are closely monitored for null values or sudden shifts."
        )
