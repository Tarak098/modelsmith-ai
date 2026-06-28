import matplotlib
matplotlib.use('Agg')  # Headless mode for web backend plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from pathlib import Path
from typing import Dict, Any
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class EDAAgent:
    def __init__(self):
        self.agent_name = "eda_agent"

    def execute(self, project_id: str, cleaned_file_path: Path, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform exploratory data analysis, generate charts, and summarize findings.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Executing Exploratory Data Analysis.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            df = pd.read_csv(cleaned_file_path)
            cols = list(df.columns)
            
            # 1. Identify Target Column
            target_col = self._identify_target(df, plan)
            log_agent_action(project_id, self.agent_name, "INFO", f"Identified target column: '{target_col}'")
            
            # 2. Basic Stats
            summary_stats = df.describe().to_dict()
            
            # 3. Numeric & Categorical Columns
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
            
            # 4. Correlation Analysis (Numeric columns only)
            correlations = {}
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                correlations = corr_matrix.to_dict()
                self._save_correlation_heatmap(project_id, corr_matrix)
                
            # 5. Save Target Distribution Chart
            self._save_target_distribution(project_id, df, target_col)
            
            # 6. Textual Insights (Simulate or use Heuristics)
            insights = self._generate_insights(df, target_col, numeric_cols, correlations)
            
            eda_report = {
                "target_column": target_col,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "summary_stats": summary_stats,
                "insights": insights,
                "charts": {
                    "correlation_heatmap": f"plots/correlation_heatmap.png",
                    "target_distribution": f"plots/target_distribution.png"
                }
            }
            
            log_agent_action(project_id, self.agent_name, "INFO", "EDA visualization plots generated and saved successfully.")
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=eda_report)
            return eda_report
            
        except Exception as e:
            error_msg = f"EDA failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _identify_target(self, df: pd.DataFrame, plan: Dict[str, Any]) -> str:
        """Identify the target column in the dataframe, leveraging LLM or heuristics."""
        cols = list(df.columns)
        desc = plan.get("description", "")
        name = plan.get("project_name", "")
        category = plan.get("category", "classification")
        
        prompt = f"""
        Given the columns of a dataset: {cols}
        And the user's machine learning goal: "{desc}" (Project: "{name}", Category: {category})
        Identify which of the column names is the target variable (the label to be predicted).
        Return ONLY the exact column name as a string, with no quotes, formatting, or extra text.
        """
        
        try:
            from auto_ai.app.infra.llm import llm_client
            target_col = llm_client.generate_text(prompt, system_instruction="You are a data architect. Return ONLY the exact column name as plain text.").strip()
            target_col = target_col.replace('"', '').replace("'", "")
            if target_col in df.columns:
                return target_col
        except Exception as e:
            from auto_ai.app.utils.logging import logger
            logger.warning(f"Failed to identify target via LLM: {e}. Falling back to heuristics.")

        # Fallback to Heuristics
        cols_lower = [c.lower() for c in df.columns]
        desc_lower = desc.lower()
        
        # Check if the description mentions any column name
        for i, col in enumerate(cols_lower):
            if col in desc_lower and col not in ["id", "index", "date", "time"]:
                return df.columns[i]
                
        standard_targets = ["outcome", "target", "price", "churn", "attrition", "label", "class", "close", "survived", "defaultrisk"]
        for target in standard_targets:
            if target in cols_lower:
                idx = cols_lower.index(target)
                return df.columns[idx]
                
        return df.columns[-1]

    def _save_correlation_heatmap(self, project_id: str, corr_matrix: pd.DataFrame):
        """Draw and save a correlation heatmap PNG."""
        plt.figure(figsize=(8, 6))
        
        # Plot correlation heatmap
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        
        # Set ticks
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(len(corr_matrix.index)), corr_matrix.index, fontsize=8)
        
        # Add labels inside cells
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.index)):
                text = f"{corr_matrix.iloc[i, j]:.2f}"
                plt.text(j, i, text, ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.6 else "white", fontsize=8)
                
        plt.title("Feature Correlation Heatmap", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        
        dest_path = StorageManager.get_plot_path(project_id, "correlation_heatmap")
        plt.savefig(dest_path, dpi=120)
        plt.close()

    def _save_target_distribution(self, project_id: str, df: pd.DataFrame, target_col: str):
        """Draw and save the target variable distribution chart."""
        plt.figure(figsize=(6, 4))
        
        target_counts = df[target_col].value_counts()
        
        # Plot appropriate chart based on column cardinality
        if len(target_counts) <= 10:
            # Bar chart for classification
            bars = plt.bar(target_counts.index.astype(str), target_counts.values, color=['#5A67D8', '#E53E3E', '#DD6B20', '#319795'][:len(target_counts)])
            plt.ylabel("Count")
            plt.xlabel(target_col)
            
            # Annotate bar counts
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        else:
            # Histogram for regression price distributions
            plt.hist(df[target_col].dropna(), bins=30, color='#3182CE', edgecolor='black', alpha=0.7)
            plt.ylabel("Frequency")
            plt.xlabel(target_col)
            
        plt.title(f"Target Distribution: {target_col}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        dest_path = StorageManager.get_plot_path(project_id, "target_distribution")
        plt.savefig(dest_path, dpi=120)
        plt.close()

    def _generate_insights(self, df: pd.DataFrame, target_col: str, numeric_cols: list, correlations: dict) -> str:
        """Create a summary description highlighting correlations and target balance."""
        summary = []
        
        # Target description
        target_counts = df[target_col].value_counts()
        if len(target_counts) <= 10:
            summary.append(f"Target '{target_col}' is categorical with distributions: ")
            for val, count in target_counts.items():
                pct = (count / len(df)) * 100
                summary.append(f"Class '{val}': {count} records ({pct:.1f}%), ")
            if len(target_counts) == 2:
                ratio = max(target_counts) / min(target_counts)
                if ratio > 2.0:
                    summary.append("Noticeable class imbalance detected. Training adjustments may be needed. ")
        else:
            mean_val = df[target_col].mean()
            std_val = df[target_col].std()
            summary.append(f"Target '{target_col}' is numerical with a mean of {mean_val:.2f} (std: {std_val:.2f}). ")
            
        # Top correlation insights
        if correlations and target_col in correlations:
            target_corrs = correlations[target_col]
            # Exclude correlation with itself
            corrs = {k: v for k, v in target_corrs.items() if k != target_col}
            if corrs:
                sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
                top_features = [f"'{f}' (r={r:.2f})" for f, r in sorted_corrs[:3]]
                summary.append(f"The features showing the strongest linear correlation to the target are: {', '.join(top_features)}.")
                
        return "".join(summary)
