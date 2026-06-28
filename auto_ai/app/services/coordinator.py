import traceback
from typing import Optional, Dict, Any
from pathlib import Path
from auto_ai.app.infra.db import update_project_record, save_agent_task_record, get_task_output_data
from auto_ai.app.utils.logging import log_agent_action, logger

# Import all agents
from auto_ai.app.services.planner import PlannerAgent
from auto_ai.app.services.data_collector import DataCollectorAgent
from auto_ai.app.services.dataset_intelligence import DatasetIntelligenceAgent, TaskConflictException
from auto_ai.app.services.automl_strategy import AutoMLStrategyAgent
from auto_ai.app.services.data_validator import DataValidatorAgent
from auto_ai.app.services.data_cleaner import DataCleanerAgent
from auto_ai.app.services.eda_agent import EDAAgent
from auto_ai.app.services.feature_engineer import FeatureEngineerAgent
from auto_ai.app.services.model_selector import ModelSelectionAgent
from auto_ai.app.services.hyperparameter_tuner import HyperparameterTunerAgent
from auto_ai.app.services.evaluator import EvaluationAgent
from auto_ai.app.services.explainability import ExplainabilityAgent
from auto_ai.app.services.report_generator import ReportGeneratorAgent
from auto_ai.app.services.memory_agent import MemoryAgent

class CoordinatorAgent:
    def __init__(self):
        self.agent_name = "coordinator"
        self.planner = PlannerAgent()
        self.collector = DataCollectorAgent()
        self.intelligence = DatasetIntelligenceAgent()
        self.strategy = AutoMLStrategyAgent()
        self.validator = DataValidatorAgent()
        self.cleaner = DataCleanerAgent()
        self.eda = EDAAgent()
        self.engineer = FeatureEngineerAgent()
        self.selector = ModelSelectionAgent()
        self.tuner = HyperparameterTunerAgent()
        self.evaluator = EvaluationAgent()
        self.explainability = ExplainabilityAgent()
        self.report_gen = ReportGeneratorAgent()
        self.memory = MemoryAgent()

    def run_workflow(self, project_id: str, name: str, description: str, uploaded_file_path: Optional[str] = None, user_selected_category: Optional[str] = None):
        """
        Main execution loop running the end-to-end ML lifecycle.
        If user_selected_category is passed, it resumes from validation.
        """
        log_agent_action(project_id, self.agent_name, "INFO", f"Initializing research workflow for project '{name}'.")
        update_project_record(project_id, status="running")
        
        agent_outputs = {}
        
        try:
            if user_selected_category:
                log_agent_action(project_id, self.agent_name, "INFO", f"Resuming workflow using user resolved task category: '{user_selected_category}'")
                
                # Fetch cached plan and raw dataset path
                plan = get_task_output_data(project_id, "planner")
                plan["category"] = user_selected_category
                agent_outputs["planner"] = plan
                
                collector_out = get_task_output_data(project_id, "data_collector")
                raw_path = Path(collector_out["raw_path"])
                agent_outputs["data_collector"] = collector_out
                
                # Re-profile intelligence and rewrite strategy with user-selected type
                update_project_record(project_id, category=user_selected_category)
                intel_report = self.intelligence.execute(project_id, plan, str(raw_path))
                intel_report["inferred_task"] = user_selected_category
                agent_outputs["dataset_intelligence"] = intel_report
                
                strategy = self.strategy.execute(project_id, intel_report)
                agent_outputs["automl_strategy"] = strategy
                
            else:
                # 1. Planner Agent
                plan = self.planner.execute(project_id, description)
                agent_outputs["planner"] = plan
                category = plan.get("category", "classification")
                update_project_record(project_id, category=category)
                
                # 2. Data Collection Agent (Retrieval + Fallback Synthetic)
                raw_path = self.collector.execute(project_id, plan, uploaded_file_path)
                agent_outputs["data_collector"] = {"raw_path": str(raw_path)}
                
                # 3. Dataset Intelligence Agent (Profiling & Conflict check)
                try:
                    intel_report = self.intelligence.execute(project_id, plan, str(raw_path))
                    agent_outputs["dataset_intelligence"] = intel_report
                except TaskConflictException as te:
                    log_agent_action(project_id, self.agent_name, "WARNING", f"Workflow execution paused: {str(te)}")
                    # Set status to awaiting_feedback and pause coordinator thread
                    update_project_record(project_id, status="awaiting_feedback", error_message=str(te))
                    return
                
                # 4. AutoML Strategy Agent
                strategy = self.strategy.execute(project_id, intel_report)
                agent_outputs["automl_strategy"] = strategy
            
            # 5. Data Validation Agent
            val_report = self.validator.execute(project_id, raw_path)
            agent_outputs["data_validator"] = val_report
            
            # 6. Data Cleaning Agent
            cleaned_path = self.cleaner.execute(project_id, raw_path, val_report)
            agent_outputs["data_cleaner"] = {"cleaned_path": str(cleaned_path)}
            
            # 7. EDA Agent
            eda_report = self.eda.execute(project_id, cleaned_path, plan)
            agent_outputs["eda_agent"] = eda_report
            
            # 8. Feature Engineering Agent
            eng_path = self.engineer.execute(project_id, cleaned_path, eda_report, plan)
            agent_outputs["feature_engineer"] = {"engineered_path": str(eng_path)}
            
            # 9. Model Selection Agent
            best_model, best_model_name, selector_summary = self.selector.execute(project_id, eng_path, eda_report, plan)
            agent_outputs["model_selector"] = selector_summary
            
            # 10. Hyperparameter Optimization Agent
            tuned_model, tuner_summary = self.tuner.execute(project_id, eng_path, eda_report, plan, best_model_name, best_model, selector_summary)
            agent_outputs["hyperparameter_tuner"] = tuner_summary
            
            # 11. Evaluation Agent
            eval_report = self.evaluator.execute(project_id, eng_path, eda_report, plan, tuned_model)
            agent_outputs["evaluator"] = eval_report
            
            # 12. Explainability Agent
            explain_report = self.explainability.execute(project_id, eng_path, eda_report, tuned_model, best_model_name)
            agent_outputs["explainability"] = explain_report
            
            # 13. Report Generation Agent
            html_report_path, md_report_path = self.report_gen.execute(project_id, name, description, agent_outputs)
            agent_outputs["report_generator"] = {
                "html_report_path": str(html_report_path),
                "md_report_path": str(md_report_path)
            }
            
            # Update category based on any model_selector target auto-correction
            category = plan.get("category", plan.get("category", "classification"))
            
            # 14. Memory Agent
            summary_metrics = {
                "description": description,
                "category": category,
                "model_name": best_model_name,
                "metrics": eval_report.get("metrics", {}),
                "tuning_metrics": tuner_summary.get("tuning_metrics", {})
            }
            self.memory.record_project_memory(project_id, name, category, summary_metrics)
            agent_outputs["memory_agent"] = {"status": "memorized"}
            
            log_agent_action(project_id, self.agent_name, "INFO", "Workflow executed successfully. Research completed.")
            update_project_record(project_id, status="completed", category=category)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Workflow error on project {project_id}: {error_trace}")
            log_agent_action(project_id, self.agent_name, "ERROR", f"Workflow execution interrupted: {str(e)}")
            update_project_record(project_id, status="failed", error_message=str(e))
