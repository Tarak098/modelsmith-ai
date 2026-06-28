import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

class ReportGeneratorAgent:
    def __init__(self):
        self.agent_name = "report_generator"

    def execute(self, project_id: str, project_name: str, description: str, all_agent_outputs: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Compile all agent outcomes and generate portable Markdown and self-contained HTML reports.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Compiling final project research report.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            # Extract outputs
            planner_data = all_agent_outputs.get("planner", {})
            validator_data = all_agent_outputs.get("data_validator", {})
            cleaner_data = all_agent_outputs.get("data_cleaner", {})
            eda_data = all_agent_outputs.get("eda_agent", {})
            fe_data = all_agent_outputs.get("feature_engineer", {})
            selector_data = all_agent_outputs.get("model_selector", {})
            tuner_data = all_agent_outputs.get("hyperparameter_tuner", {})
            eval_data = all_agent_outputs.get("evaluator", {})
            exp_data = all_agent_outputs.get("explainability", {})
            
            run_dir = StorageManager.get_run_dir(project_id)
            
            # Base64 encode plots for self-contained HTML rendering
            corr_img = self._get_base64_image(run_dir / "plots" / "correlation_heatmap.png")
            dist_img = self._get_base64_image(run_dir / "plots" / "target_distribution.png")
            
            metric_plot_name = "confusion_matrix.png" if planner_data.get("category") == "classification" else "residual_plot.png"
            metric_img = self._get_base64_image(run_dir / "plots" / metric_plot_name)
            
            imp_img = self._get_base64_image(run_dir / "plots" / "feature_importance.png")
            
            # 1. Generate HTML Report
            html_content = self._build_html_report(
                project_name, description, planner_data, validator_data, cleaner_data,
                eda_data, fe_data, selector_data, tuner_data, eval_data, exp_data,
                corr_img, dist_img, metric_img, imp_img
            )
            
            # 2. Generate Markdown Report
            md_content = self._build_markdown_report(
                project_name, description, planner_data, validator_data, cleaner_data,
                eda_data, fe_data, selector_data, tuner_data, eval_data, exp_data
            )
            
            # Save files
            html_path = StorageManager.save_report(project_id, html_content, "report.html")
            md_path = StorageManager.save_report(project_id, md_content, "report.md")
            
            summary = {
                "html_report_path": str(html_path),
                "markdown_report_path": str(md_path)
            }
            
            log_agent_action(project_id, self.agent_name, "INFO", "Reports compiled and saved to disk.")
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=summary)
            return html_path, md_path
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _get_base64_image(self, path: Path) -> str:
        """Read an image file and encode it as a base64 Data URI."""
        if path.exists():
            try:
                with open(path, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode('utf-8')
                    return f"data:image/png;base64,{encoded}"
            except Exception as e:
                logger.error(f"Failed to base64 encode {path}: {e}")
        return ""

    def _build_html_report(self, name: str, desc: str, planner: dict, validator: dict, cleaner: dict, eda: dict, fe: dict, selector: dict, tuner: dict, evaluation: dict, explain: dict, corr_img: str, dist_img: str, eval_img: str, imp_img: str) -> str:
        # Create HTML string
        category = planner.get("category", "N/A")
        
        # Build metrics rows
        metrics_html = ""
        for k, v in evaluation.get("metrics", {}).items():
            val_str = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            metrics_html += f"<tr><td><strong>{k.replace('_', ' ').title()}</strong></td><td>{val_str}</td></tr>"
            
        # Leaderboard rows
        leaderboard_html = ""
        for idx, row in enumerate(selector.get("leaderboard", [])):
            metrics_str = ", ".join(f"{mk}: {mv:.3f}" for mk, mv in row.get("metrics", {}).items())
            leaderboard_html += f"<tr><td>{idx+1}</td><td>{row.get('model_name')}</td><td>{metrics_str}</td></tr>"

        # Build diagnostics list HTML
        diagnoses_html = ""
        for diag in evaluation.get('diagnoses', []):
            css_class = "diag-warn" if "WARNING" in diag else "diag-success"
            diagnoses_html += f'<div class="{css_class}">{diag}</div>'
            
        # Build limitations list HTML
        limitations_html = ""
        for lim in explain.get('limitations', []):
            limitations_html += f"<li>{lim}</li>"

        # Tuning improvements
        tuning_html = ""
        t_metrics = tuner.get("tuning_metrics", {})
        if t_metrics:
            tuning_html = f"""
            <p><strong>Hyperparameter optimization:</strong></p>
            <ul>
                <li>Parameters tuned: <code>{tuner.get('tuned_parameters', {})}</code></li>
                <li>Performance before tuning: <code>{t_metrics.get('before_tune', 0):.4f}</code></li>
                <li>Performance after tuning: <code>{t_metrics.get('after_tune', 0):.4f}</code></li>
                <li>Relative Improvement: <code>{tuner.get('improvement_ratio', 0)*100:+.2f}%</code></li>
            </ul>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ModelSmith AI Report - {name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        h1 {{
            color: #818cf8;
            font-size: 2.5em;
            margin-top: 0;
            border-bottom: 2px solid rgba(129, 140, 248, 0.3);
            padding-bottom: 15px;
        }}
        h2 {{
            color: #6366f1;
            margin-top: 30px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }}
        th {{
            background-color: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
            background: rgba(15, 23, 42, 0.4);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .badge {{
            background-color: #3b82f6;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.9em;
            display: inline-block;
            margin-bottom: 15px;
        }}
        .diag-warn {{
            background: rgba(239, 68, 68, 0.15);
            border-left: 4px solid #ef4444;
            color: #fca5a5;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .diag-success {{
            background: rgba(16, 185, 129, 0.15);
            border-left: 4px solid #10b981;
            color: #a7f3d0;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ModelSmith AI Research Report</h1>
        <div class="badge">Category: {category.upper()}</div>
        
        <h2>Project Overview</h2>
        <p><strong>Project Name:</strong> {name}</p>
        <p><strong>Research Description:</strong> {desc}</p>
        
        <h2>Data Validation & Cleaning Summary</h2>
        <div class="grid">
            <div class="card">
                <h3>Raw Profile</h3>
                <ul>
                    <li>Rows detected: {validator.get('num_rows', 0)}</li>
                    <li>Columns detected: {validator.get('num_cols', 0)}</li>
                    <li>Duplicates: {validator.get('duplicate_count', 0)}</li>
                    <li>Quality Score: {validator.get('data_quality_score', 0)}/100</li>
                </ul>
            </div>
            <div class="card">
                <h3>Cleaning Action</h3>
                <ul>
                    <li>Deduplicated rows: {cleaner.get('removed_duplicates', 0)}</li>
                    <li>Final row count: {cleaner.get('cleaned_rows', 0)}</li>
                    <li>Imputation applied: {len(cleaner.get('imputations', {}))} columns</li>
                </ul>
            </div>
        </div>

        <h2>Exploratory Data Analysis</h2>
        <p>{eda.get('insights', 'N/A')}</p>
        <div class="grid">
            <div class="plot-container">
                <h4>Correlation Analysis</h4>
                {f'<img src="{corr_img}">' if corr_img else '<p>No image generated</p>'}
            </div>
            <div class="plot-container">
                <h4>Target Distribution</h4>
                {f'<img src="{dist_img}">' if dist_img else '<p>No image generated</p>'}
            </div>
        </div>

        <h2>Model Selection & Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model Name</th>
                    <th>Validation Metrics</th>
                </tr>
            </thead>
            <tbody>
                {leaderboard_html}
            </tbody>
        </table>
        
        {tuning_html}

        <h2>Final Evaluation Results</h2>
        <div class="grid">
            <div class="card">
                <h3>Metrics</h3>
                <table>
                    {metrics_html}
                </table>
            </div>
            <div class="card">
                <h3>Diagnostics</h3>
                {diagnoses_html}
            </div>
        </div>
        
        <div class="grid">
            <div class="plot-container">
                <h4>Model Performance Metrics</h4>
                {f'<img src="{eval_img}">' if eval_img else '<p>No image generated</p>'}
            </div>
            <div class="plot-container">
                <h4>Predictive Feature Importance</h4>
                {f'<img src="{imp_img}">' if imp_img else '<p>No image generated</p>'}
            </div>
        </div>

        <h2>Model Explainability</h2>
        <p><strong>Natural Language Interpretation:</strong></p>
        <p>{explain.get('natural_language_explanation', 'N/A')}</p>
        
        <p><strong>Identified Model Constraints:</strong></p>
        <ul>
            {limitations_html}
        </ul>
        
        <footer style="margin-top: 40px; text-align: center; font-size: 0.8em; color: #64748b;">
            Report compiled by ModelSmith AI Agent Platform. All rights reserved.
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _build_markdown_report(self, name: str, desc: str, planner: dict, validator: dict, cleaner: dict, eda: dict, fe: dict, selector: dict, tuner: dict, evaluation: dict, explain: dict) -> str:
        category = planner.get("category", "N/A")
        
        # Leaderboard
        leaderboard_md = "| Rank | Model Name | Validation Metrics |\n|---|---|---|\n"
        for idx, row in enumerate(selector.get("leaderboard", [])):
            metrics_str = ", ".join(f"{mk}: {mv:.3f}" for mk, mv in row.get("metrics", {}).items())
            leaderboard_md += f"| {idx+1} | {row.get('model_name')} | {metrics_str} |\n"
            
        # Metrics
        metrics_md = ""
        for k, v in evaluation.get("metrics", {}).items():
            val_str = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            metrics_md += f"- **{k.replace('_', ' ').title()}**: {val_str}\n"

        # Pre-render diagnostics and limitations markdown lists (avoid backslashes in f-string expression)
        diagnoses_md = "\n".join(f"- {diag}" for diag in evaluation.get('diagnoses', []))
        limitations_md = "\n".join(f"- {lim}" for lim in explain.get('limitations', []))

        md = f"""# ModelSmith AI Research Report

**Category:** {category.upper()}
**Project Name:** {name}
**Description:** {desc}

---

## Data Summary
- **Raw Row Count:** {validator.get('num_rows', 0)}
- **Clean Row Count:** {cleaner.get('cleaned_rows', 0)}
- **Duplicates Removed:** {cleaner.get('removed_duplicates', 0)}
- **Data Quality Score:** {validator.get('data_quality_score', 0)}/100

### EDA Insights
{eda.get('insights', 'N/A')}

---

## Model Selection Leaderboard
{leaderboard_md}

### Tuning Optimization
- **Parameters:** {tuner.get('tuned_parameters', {})}
- **Relative Score Improvement:** {tuner.get('improvement_ratio', 0)*100:+.2f}%

---

## Final Performance & Diagnostics
{metrics_md}

### Fit Diagnostics
{diagnoses_md}

---

## Model Interpretability & Explainability
{explain.get('natural_language_explanation', 'N/A')}

### Model Limitations
{limitations_md}

---
Report generated autonomously by ModelSmith AI.
"""
        return md
