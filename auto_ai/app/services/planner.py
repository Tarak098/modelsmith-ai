import json
from typing import Dict, Any
from auto_ai.app.infra.llm import llm_client
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.core.security import scan_prompt_injection

class PlannerAgent:
    def __init__(self):
        self.agent_name = "planner"

    def execute(self, project_id: str, description: str) -> Dict[str, Any]:
        """
        Analyze user's ML request description and create an execution plan.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Planner starting analysis of user request.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            # Check for security issues first
            sanitized_desc = scan_prompt_injection(description)
            
            # Formulate detailed prompt for LLM
            prompt = f"""
            You are the lead Planner Agent for ModelSmith AI.
            Analyze the following natural language request for a machine learning task:
            "{sanitized_desc}"
            
            Classify this task and generate a structured JSON execution plan containing:
            1. "project_name": A short descriptive name for this research.
            2. "category": Choose exactly one from ["classification", "regression", "forecasting", "clustering", "nlp", "cv"].
            3. "model_candidates": A list of 3-4 suitable scikit-learn/ML model algorithms to try.
            4. "resource_estimate": An object estimating CPU, Memory, and expected execution speed.
            5. "steps": An array of downstream agent names to activate. Must list:
               ["data_collector", "data_validator", "data_cleaner", "eda_agent", "feature_engineer", "model_selector", "hyperparameter_tuner", "evaluator", "explainability", "report_generator", "memory_agent"]
            6. "validation_checks": 2 validation items ensuring dataset readiness.
            
            Return ONLY a raw JSON object, without markdown formatting.
            """
            
            # Execute LLM call
            plan = llm_client.generate_json(prompt, system_instruction="You are a data science architect. Respond only with raw JSON.")
            plan["description"] = description
            
            category = plan.get("category", "classification")
            project_name = plan.get("project_name", "ML Research Project")
            
            log_agent_action(
                project_id, 
                self.agent_name, 
                "INFO", 
                f"Request classified. Name: '{project_name}', Category: '{category}'. Workflow assembled with {len(plan.get('steps', []))} steps."
            )
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data=plan)
            return plan
            
        except Exception as e:
            error_msg = f"Planner failed: {e}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise e
