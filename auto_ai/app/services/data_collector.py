import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from auto_ai.app.infra.storage import StorageManager
from auto_ai.app.infra.db import save_agent_task_record
from auto_ai.app.utils.logging import log_agent_action
from auto_ai.app.utils.exceptions import AgentException

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
                
            # Case 2: No file provided -> Generate Domain-Specific Synthetic Dataset
            log_agent_action(project_id, self.agent_name, "WARNING", "No dataset uploaded. Generating realistic synthetic dataset for ML execution.")
            
            category = plan.get("category", "classification")
            name = plan.get("project_name", "").lower()
            
            df = self._generate_synthetic_data(name, category)
            
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

    def _generate_synthetic_data(self, name: str, category: str) -> pd.DataFrame:
        """Create realistic domain-specific tabular datasets."""
        np.random.seed(42)
        n_samples = 500
        
        # 1. Diabetes Prediction
        if "diabet" in name:
            data = {
                "Pregnancies": np.random.randint(0, 15, size=n_samples),
                "Glucose": np.random.randint(70, 200, size=n_samples),
                "BloodPressure": np.random.randint(60, 110, size=n_samples),
                "SkinThickness": np.random.randint(10, 50, size=n_samples),
                "Insulin": np.random.randint(15, 300, size=n_samples),
                "BMI": np.round(np.random.uniform(18.0, 45.0, size=n_samples), 1),
                "DiabetesPedigree": np.round(np.random.uniform(0.1, 2.0, size=n_samples), 3),
                "Age": np.random.randint(21, 80, size=n_samples),
            }
            # Inject some missing values (NaN) to test validation & cleaning
            for col in ["BloodPressure", "Insulin", "SkinThickness"]:
                mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
                data[col] = np.where(mask, np.nan, data[col])
                
            # Outcome correlation with Glucose and BMI
            df = pd.DataFrame(data)
            score = (df["Glucose"] * 0.05) + (df["BMI"] * 0.1) - 10
            probs = 1 / (1 + np.exp(-score))
            df["Outcome"] = np.random.binomial(1, probs)
            return df
            
        # 2. House Prices Regression
        elif "house" in name or "price" in name:
            sqft = np.random.randint(800, 5000, size=n_samples)
            bedrooms = np.random.randint(1, 6, size=n_samples)
            bathrooms = bedrooms + np.random.randint(0, 2, size=n_samples)
            year_built = np.random.randint(1950, 2025, size=n_samples)
            neighborhood = np.random.choice(["Downtown", "Suburbs", "Uptown", "Rural"], size=n_samples)
            
            data = {
                "SquareFeet": sqft,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "YearBuilt": year_built,
                "Neighborhood": neighborhood,
                "GarageSize": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.3, 0.5, 0.1])
            }
            # Add NaNs
            mask = np.random.choice([True, False], size=n_samples, p=[0.04, 0.96])
            data["GarageSize"] = np.where(mask, np.nan, data["GarageSize"])
            
            df = pd.DataFrame(data)
            # Neighborhood weights
            nb_weights = {"Downtown": 50000, "Suburbs": 30000, "Uptown": 75000, "Rural": -20000}
            nb_add = df["Neighborhood"].map(nb_weights)
            
            # Base price formula
            price = 50000 + (df["SquareFeet"] * 120) + (df["Bedrooms"] * 15000) + (df["Bathrooms"] * 10000) + ((df["YearBuilt"] - 1950) * 1000) + nb_add
            price += np.random.normal(0, 15000, size=n_samples)
            df["Price"] = np.round(price, -2)
            return df

        # 3. Customer Churn
        elif "churn" in name:
            data = {
                "TenureMonths": np.random.randint(1, 72, size=n_samples),
                "MonthlyCharges": np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2),
                "ContractType": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.5, 0.3, 0.2]),
                "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.4, 0.2]),
                "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples),
                "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples)
            }
            df = pd.DataFrame(data)
            df["TotalCharges"] = np.round(df["TenureMonths"] * df["MonthlyCharges"] + np.random.normal(0, 10, size=n_samples), 2)
            
            # Formulate churn correlation
            score = (df["ContractType"] == "Month-to-month").astype(int) * 2.0 + (df["InternetService"] == "Fiber optic").astype(int) * 1.0 - (df["TenureMonths"] * 0.05)
            probs = 1 / (1 + np.exp(-score))
            df["Churn"] = np.random.binomial(1, probs)
            return df
            
        # 4. Default Mock Dataset (for classification or regression fallback)
        else:
            data = {
                "Feature_1": np.random.normal(10, 2, size=n_samples),
                "Feature_2": np.random.uniform(0, 100, size=n_samples),
                "Feature_3": np.random.exponential(5, size=n_samples),
                "Category_1": np.random.choice(["Low", "Medium", "High"], size=n_samples)
            }
            # Missing columns
            mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
            data["Feature_1"] = np.where(mask, np.nan, data["Feature_1"])
            
            df = pd.DataFrame(data)
            if category == "regression":
                df["Target"] = df["Feature_1"] * 2.5 + (df["Category_1"].map({"Low": 1, "Medium": 5, "High": 10}) * 3.0) + np.random.normal(0, 1, size=n_samples)
            else:
                score = df["Feature_1"] * 0.2 - df["Feature_2"] * 0.05
                df["Target"] = np.where(score > np.median(score), 1, 0)
                
            return df
