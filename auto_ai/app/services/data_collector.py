import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException
from auto_ai.app.services.dataset_retriever import DatasetRetrievalAgent

class DataCollectorAgent:
    def __init__(self):
        self.agent_name = "data_collector"

    def execute(self, project_id: str, plan: Dict[str, Any], uploaded_file_path: Optional[str] = None) -> Path:
        """
        Gathers or generates raw dataset for training.
        """
        log_agent_action(project_id, self.agent_name, "INFO", "Starting data acquisition step.")
        save_agent_task_record(project_id, self.agent_name, "running")
        
        try:
            # Case 1: User uploaded a file
            if uploaded_file_path and Path(uploaded_file_path).exists():
                src_path = Path(uploaded_file_path)
                dest_filename = f"raw_data{src_path.suffix}"
                dest_path = StorageManager.copy_file_to_run(project_id, src_path, dest_filename)
                log_agent_action(project_id, self.agent_name, "INFO", f"Acquired user-uploaded dataset from {uploaded_file_path}.")
                
                # Basic validation that it loads
                df = pd.read_csv(dest_path)
                save_agent_task_record(project_id, self.agent_name, "completed", output_data={
                    "source": "upload",
                    "rows": len(df),
                    "cols": len(df.columns),
                    "columns": list(df.columns)
                })
                return dest_path
                
            # Case 2: Attempt public repository dataset retrieval first
            desc = plan.get("description", "")
            category = plan.get("category", "classification")
            name = plan.get("project_name", "")
            
            # Formulate a simplified search keyword query from LLM-generated project name
            search_query = name if name else desc
            query_clean = search_query.lower()
            for stop_word in ["prediction", "classification", "forecasting", "regression", "task", "analysis", "dataset", "study", "project", "model", "generic"]:
                query_clean = query_clean.replace(stop_word, "")
            query_clean = " ".join(query_clean.split()).strip()
            if not query_clean:
                query_clean = "dataset"
                
            log_agent_action(project_id, self.agent_name, "INFO", f"Searching public repositories sequentially for topic: '{query_clean}'...")
            retriever = DatasetRetrievalAgent()
            retrieved_path = retriever.execute(query_clean, expected_task=category)
            
            if retrieved_path and Path(retrieved_path).exists():
                src_path = Path(retrieved_path)
                dest_filename = "raw_data.csv"
                dest_path = StorageManager.copy_file_to_run(project_id, src_path, dest_filename)
                log_agent_action(project_id, self.agent_name, "INFO", f"Successfully retrieved public dataset for '{desc}'. Source: Cached/Downloaded CSV.")
                
                df = pd.read_csv(dest_path)
                save_agent_task_record(project_id, self.agent_name, "completed", output_data={
                    "source": "retrieved_public_repository",
                    "rows": len(df),
                    "cols": len(df.columns),
                    "columns": list(df.columns)
                })
                return dest_path
                
            # Case 3: Fallback -> Generate Domain-Specific Synthetic Dataset
            log_agent_action(project_id, self.agent_name, "WARNING", "No public dataset found above suitability score. Generating realistic synthetic dataset.")
            
            df = self._generate_synthetic_data(project_id, name, category, desc)
            dest_path = StorageManager.save_dataset(project_id, df, "raw_data.csv")
            
            log_agent_action(
                project_id, 
                self.agent_name, 
                "INFO", 
                f"Generated synthetic '{category}' dataset containing {len(df)} rows and {len(df.columns)} columns."
            )
            
            save_agent_task_record(project_id, self.agent_name, "completed", output_data={
                "source": "synthetic_generator",
                "rows": len(df),
                "cols": len(df.columns),
                "columns": list(df.columns)
            })
            return dest_path
            
        except Exception as e:
            error_msg = f"Data Collection failed: {str(e)}"
            log_agent_action(project_id, self.agent_name, "ERROR", error_msg)
            save_agent_task_record(project_id, self.agent_name, "failed", output_data={"error": error_msg})
            raise AgentException(self.agent_name, error_msg)

    def _generate_synthetic_data(self, project_id: str, name: str, category: str, desc: str) -> pd.DataFrame:
        """Create realistic domain-specific tabular datasets dynamically using a schema."""
        # Seeding based on project_id characters ensures each run is unique but repeatable if needed.
        seed_val = sum(ord(c) for c in project_id)
        np.random.seed(seed_val % 10000)
        n_samples = 500

        # Retrieve/build the schema based on description or category
        schema = self._get_dataset_schema(category, name, desc)
        
        # Build the dataset using the schema
        return self._generate_data_from_schema(schema, n_samples)

    def _get_dataset_schema(self, category: str, name: str, desc: str) -> Dict[str, Any]:
        """Gets the dataset schema via LLM or rule-based fallback."""
        prompt = f"""
        Design a realistic synthetic dataset schema for the following machine learning prompt:
        Prompt: "{desc}"
        Project Name: "{name}"
        Category: {category}

        Generate a JSON object containing:
        1. "target_column": Name of the target variable (no spaces, e.g. "HeartDisease", "CarPrice").
        2. "target_type": "binary" (classification) or "continuous" (regression) or "multiclass".
        3. "features": A list of 5-7 feature columns, each containing:
           - "name": Name of the feature (no spaces, e.g. "Age", "Income").
           - "type": "numeric" or "categorical".
           - "min_val": (only if numeric) Minimum value.
           - "max_val": (only if numeric) Maximum value.
           - "categories": (only if categorical) List of string categories.
           - "correlation_weight": A float between -1.0 and 1.0 indicating how it correlates to target.
        """
        try:
            from auto_ai.app.infra.llm import llm_client
            # Request LLM JSON
            schema = llm_client.generate_json(prompt, system_instruction="You are a data science architect. Respond only with raw JSON.")
            if schema and "target_column" in schema and "features" in schema:
                return schema
        except Exception as e:
            from auto_ai.app.utils.logging import logger
            logger.warning(f"Failed to get dynamic schema from LLM: {e}. Falling back to Rule-Based Heuristic schemas.")

        # Fallback to Rule-Based Heuristics
        return self._get_fallback_dataset_schema(category, name, desc)

    def _get_fallback_dataset_schema(self, category: str, name: str, desc: str) -> Dict[str, Any]:
        """Heuristic rule-based database of domain schemas."""
        text = f"{name} {desc}".lower()
        
        # 1. Diabetes
        if "diabet" in text:
            return {
                "target_column": "Outcome",
                "target_type": "binary",
                "features": [
                    {"name": "Pregnancies", "type": "numeric", "min_val": 0, "max_val": 15, "correlation_weight": 0.2},
                    {"name": "Glucose", "type": "numeric", "min_val": 70, "max_val": 200, "correlation_weight": 0.8},
                    {"name": "BloodPressure", "type": "numeric", "min_val": 60, "max_val": 110, "correlation_weight": 0.1},
                    {"name": "BMI", "type": "numeric", "min_val": 18.0, "max_val": 45.0, "correlation_weight": 0.5},
                    {"name": "DiabetesPedigree", "type": "numeric", "min_val": 0.1, "max_val": 2.0, "correlation_weight": 0.3},
                    {"name": "Age", "type": "numeric", "min_val": 21, "max_val": 80, "correlation_weight": 0.4}
                ]
            }
            
        # 2. House Prices
        if any(k in text for k in ["house", "home", "rent", "estate", "apartment"]):
            return {
                "target_column": "Price",
                "target_type": "continuous",
                "features": [
                    {"name": "SquareFeet", "type": "numeric", "min_val": 800, "max_val": 5000, "correlation_weight": 0.8},
                    {"name": "Bedrooms", "type": "numeric", "min_val": 1, "max_val": 5, "correlation_weight": 0.4},
                    {"name": "Bathrooms", "type": "numeric", "min_val": 1, "max_val": 4, "correlation_weight": 0.5},
                    {"name": "YearBuilt", "type": "numeric", "min_val": 1950, "max_val": 2025, "correlation_weight": 0.3},
                    {"name": "Neighborhood", "type": "categorical", "categories": ["Downtown", "Suburbs", "Uptown", "Rural"], "correlation_weight": 0.4}
                ]
            }

        # 3. Car Price
        if any(k in text for k in ["car", "auto", "vehicle", "motor"]):
            return {
                "target_column": "Price",
                "target_type": "continuous",
                "features": [
                    {"name": "EngineSize", "type": "numeric", "min_val": 1.0, "max_val": 6.0, "correlation_weight": 0.6},
                    {"name": "Horsepower", "type": "numeric", "min_val": 80, "max_val": 500, "correlation_weight": 0.8},
                    {"name": "Year", "type": "numeric", "min_val": 2005, "max_val": 2025, "correlation_weight": 0.4},
                    {"name": "Mileage", "type": "numeric", "min_val": 5000, "max_val": 200000, "correlation_weight": -0.7},
                    {"name": "Transmission", "type": "categorical", "categories": ["Automatic", "Manual"], "correlation_weight": -0.2}
                ]
            }

        # 4. Customer Churn
        if any(k in text for k in ["churn", "attrition", "loyalty", "subscriber"]):
            return {
                "target_column": "Churn",
                "target_type": "binary",
                "features": [
                    {"name": "TenureMonths", "type": "numeric", "min_val": 1, "max_val": 72, "correlation_weight": -0.6},
                    {"name": "MonthlyCharges", "type": "numeric", "min_val": 20, "max_val": 120, "correlation_weight": 0.4},
                    {"name": "SupportCalls", "type": "numeric", "min_val": 0, "max_val": 10, "correlation_weight": 0.7},
                    {"name": "ContractType", "type": "categorical", "categories": ["Month-to-month", "One year", "Two year"], "correlation_weight": -0.8},
                    {"name": "InternetService", "type": "categorical", "categories": ["DSL", "Fiber optic", "No"], "correlation_weight": 0.5}
                ]
            }

        # 5. Credit Risk
        if any(k in text for k in ["credit", "default", "loan", "risk", "bank", "finance"]):
            return {
                "target_column": "DefaultRisk",
                "target_type": "binary",
                "features": [
                    {"name": "Age", "type": "numeric", "min_val": 18, "max_val": 70, "correlation_weight": -0.2},
                    {"name": "AnnualIncome", "type": "numeric", "min_val": 20000, "max_val": 150000, "correlation_weight": -0.5},
                    {"name": "CreditScore", "type": "numeric", "min_val": 300, "max_val": 850, "correlation_weight": -0.8},
                    {"name": "DebtToIncomeRatio", "type": "numeric", "min_val": 0.05, "max_val": 0.95, "correlation_weight": 0.7},
                    {"name": "EmploymentYears", "type": "numeric", "min_val": 0, "max_val": 40, "correlation_weight": -0.4}
                ]
            }

        # 6. Salary Prediction
        if any(k in text for k in ["salary", "income", "earnings", "wage", "pay"]):
            return {
                "target_column": "Salary",
                "target_type": "continuous",
                "features": [
                    {"name": "YearsExperience", "type": "numeric", "min_val": 0, "max_val": 40, "correlation_weight": 0.9},
                    {"name": "EducationYears", "type": "numeric", "min_val": 12, "max_val": 22, "correlation_weight": 0.6},
                    {"name": "ManagementRole", "type": "categorical", "categories": ["Yes", "No"], "correlation_weight": 0.5},
                    {"name": "CompanySize", "type": "categorical", "categories": ["Startup", "Mid-size", "Enterprise"], "correlation_weight": 0.3}
                ]
            }

        # 7. Student Grades
        if any(k in text for k in ["student", "grade", "score", "exam", "education", "school"]):
            return {
                "target_column": "FinalGrade",
                "target_type": "continuous",
                "features": [
                    {"name": "StudyHours", "type": "numeric", "min_val": 0, "max_val": 40, "correlation_weight": 0.8},
                    {"name": "AttendanceRate", "type": "numeric", "min_val": 0.5, "max_val": 1.0, "correlation_weight": 0.7},
                    {"name": "SleepHours", "type": "numeric", "min_val": 4, "max_val": 10, "correlation_weight": 0.3},
                    {"name": "ParentalSupport", "type": "categorical", "categories": ["Low", "Medium", "High"], "correlation_weight": 0.4}
                ]
            }

        # 8. Crop Yield
        if any(k in text for k in ["crop", "yield", "farm", "agriculture", "plant"]):
            return {
                "target_column": "Yield",
                "target_type": "continuous",
                "features": [
                    {"name": "Rainfall", "type": "numeric", "min_val": 10, "max_val": 200, "correlation_weight": 0.6},
                    {"name": "Temperature", "type": "numeric", "min_val": 15, "max_val": 40, "correlation_weight": 0.4},
                    {"name": "SoilQualityScore", "type": "numeric", "min_val": 10, "max_val": 100, "correlation_weight": 0.8},
                    {"name": "FertilizerType", "type": "categorical", "categories": ["Organic", "Chemical", "None"], "correlation_weight": 0.3}
                ]
            }

        # 9. Weather Temperature
        if any(k in text for k in ["weather", "temperature", "temp", "rain", "humidity"]):
            return {
                "target_column": "Temperature",
                "target_type": "continuous",
                "features": [
                    {"name": "Date", "type": "categorical", "categories": ["DatePlaceHolder"], "correlation_weight": 0.0},
                    {"name": "Humidity", "type": "numeric", "min_val": 0.1, "max_val": 1.0, "correlation_weight": -0.6},
                    {"name": "WindSpeed", "type": "numeric", "min_val": 0, "max_val": 50, "correlation_weight": -0.3},
                    {"name": "Pressure", "type": "numeric", "min_val": 980, "max_val": 1030, "correlation_weight": 0.2}
                ]
            }

        # 10. Web Traffic
        if any(k in text for k in ["traffic", "view", "visitor", "website", "clicks"]):
            return {
                "target_column": "PageViews",
                "target_type": "continuous",
                "features": [
                    {"name": "Date", "type": "categorical", "categories": ["DatePlaceHolder"], "correlation_weight": 0.0},
                    {"name": "MarketingSpend", "type": "numeric", "min_val": 0, "max_val": 5000, "correlation_weight": 0.8},
                    {"name": "PromoActive", "type": "categorical", "categories": ["Yes", "No"], "correlation_weight": 0.4}
                ]
            }

        # 11. Store Sales
        if any(k in text for k in ["sale", "revenue", "shop", "retail", "store"]):
            return {
                "target_column": "Sales",
                "target_type": "continuous",
                "features": [
                    {"name": "Date", "type": "categorical", "categories": ["DatePlaceHolder"], "correlation_weight": 0.0},
                    {"name": "PromoActive", "type": "categorical", "categories": ["Yes", "No"], "correlation_weight": 0.5},
                    {"name": "CompetitorDistance", "type": "numeric", "min_val": 100, "max_val": 15000, "correlation_weight": -0.4}
                ]
            }

        # 12. Dynamic Fallback by Category
        if category == "forecasting":
            return {
                "target_column": "QuantityDemand",
                "target_type": "continuous",
                "features": [
                    {"name": "Date", "type": "categorical", "categories": ["DatePlaceHolder"], "correlation_weight": 0.0},
                    {"name": "UnitRate", "type": "numeric", "min_val": 5, "max_val": 200, "correlation_weight": -0.5},
                    {"name": "IsPromoWeekend", "type": "categorical", "categories": ["Yes", "No"], "correlation_weight": 0.6}
                ]
            }
        elif category == "regression":
            return {
                "target_column": "ValueTarget",
                "target_type": "continuous",
                "features": [
                    {"name": "FeatureSize", "type": "numeric", "min_val": 1, "max_val": 1000, "correlation_weight": 0.7},
                    {"name": "FeatureQuality", "type": "numeric", "min_val": 0.1, "max_val": 9.9, "correlation_weight": 0.6},
                    {"name": "FeatureRating", "type": "numeric", "min_val": 0, "max_val": 5, "correlation_weight": 0.4},
                    {"name": "FeatureClass", "type": "categorical", "categories": ["Economy", "Standard", "Premium"], "correlation_weight": 0.5}
                ]
            }
        else:  # classification default
            return {
                "target_column": "ClassLabel",
                "target_type": "binary",
                "features": [
                    {"name": "FeatureAge", "type": "numeric", "min_val": 18, "max_val": 90, "correlation_weight": 0.3},
                    {"name": "FeatureScore", "type": "numeric", "min_val": 0.0, "max_val": 100.0, "correlation_weight": 0.6},
                    {"name": "FeatureCategory", "type": "categorical", "categories": ["GroupA", "GroupB", "GroupC"], "correlation_weight": -0.4}
                ]
            }

    def _generate_data_from_schema(self, schema: Dict[str, Any], n_samples: int = 500) -> pd.DataFrame:
        """Programmatically generate a dataset matching the given schema layout."""
        data = {}
        target_score = np.zeros(n_samples)
        
        for feat in schema.get("features", []):
            name = feat["name"]
            f_type = feat["type"]
            weight = feat.get("correlation_weight", 0.0)
            
            # Special case for Date
            if name.lower() == "date":
                dates = pd.date_range(start="2026-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d").tolist()
                data[name] = dates
                continue
                
            if f_type == "numeric":
                min_v = feat.get("min_val", 0.0)
                max_v = feat.get("max_val", 100.0)
                vals = np.random.uniform(min_v, max_v, size=n_samples)
                data[name] = np.round(vals, 1 if max_v - min_v < 10 else 0)
                
                mean_v = (max_v + min_v) / 2
                range_v = (max_v - min_v) / 2 if max_v != min_v else 1.0
                target_score += ((vals - mean_v) / range_v) * weight
                
            elif f_type == "categorical":
                cats = feat.get("categories", ["Low", "Medium", "High"])
                if not cats or (len(cats) == 1 and cats[0] == "DatePlaceHolder"):
                    cats = ["Low", "Medium", "High"]
                vals = np.random.choice(cats, size=n_samples)
                data[name] = vals
                
                mapped = np.array([cats.index(v) for v in vals])
                mean_v = (len(cats) - 1) / 2
                range_v = (len(cats) - 1) / 2 if len(cats) > 1 else 1.0
                target_score += ((mapped - mean_v) / range_v) * weight
                
        target_name = schema.get("target_column", "Target")
        target_type = schema.get("target_type", "binary")
        
        if not target_name:
            target_name = "Target"
            
        if target_type == "binary":
            probs = 1 / (1 + np.exp(-target_score))
            data[target_name] = np.random.binomial(1, probs)
        else:
            noise = np.random.normal(0, 1.0, size=n_samples)
            data[target_name] = np.round(200 + (target_score + noise) * 50, 1)
            
        # Basic sanity check: inject some missing values to keep cleaner/validator testable
        df = pd.DataFrame(data)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if target_name in num_cols:
            num_cols.remove(target_name)
        if num_cols:
            target_col_to_nan = num_cols[0]
            mask = np.random.choice([True, False], size=n_samples, p=[0.03, 0.97])
            df[target_col_to_nan] = np.where(mask, np.nan, df[target_col_to_nan])
            
        return df
