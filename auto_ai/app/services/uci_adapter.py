import urllib.request
import logging
import pandas as pd
from typing import List, Dict, Any
from auto_ai.app.services.repository_adapter import RepositoryAdapter

logger = logging.getLogger(__name__)

UCI_DATASETS = [
    {
        "name": "Wine Quality",
        "keywords": ["wine", "quality", "alcohol", "red wine", "white wine"],
        "description": "Wine Quality Dataset containing physicochemical properties of red Portuguese 'Vinho Verde' wine.",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "rows": 1599,
        "features": 12,
        "target": "quality",
        "popularity": 0.95,
        "license": "CC0",
        "file_size_mb": 0.1,
        "columns": None
    },
    {
        "name": "Car Evaluation",
        "keywords": ["car", "vehicle", "evaluation", "buying", "safety"],
        "description": "Car Evaluation Database derived from a simple hierarchical decision model.",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        "rows": 1728,
        "features": 7,
        "target": "class",
        "popularity": 0.9,
        "license": "CC-BY-4.0",
        "file_size_mb": 0.05,
        "columns": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    },
    {
        "name": "Iris Dataset",
        "keywords": ["iris", "flower", "setosa", "versicolor", "virginica"],
        "description": "Iris plants dataset. Classification of three classes of iris flowers.",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "rows": 150,
        "features": 5,
        "target": "class",
        "popularity": 1.0,
        "license": "CC0",
        "file_size_mb": 0.01,
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    },
    {
        "name": "Heart Disease Cleveland",
        "keywords": ["heart", "disease", "cardiovascular", "cleveland", "medical"],
        "description": "Heart disease database containing patient clinical metrics from Cleveland Clinic.",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "rows": 303,
        "features": 14,
        "target": "num",
        "popularity": 0.9,
        "license": "CC-BY",
        "file_size_mb": 0.02,
        "columns": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    },
    {
        "name": "Adult Income",
        "keywords": ["adult", "income", "census", "salary", "age", "wage"],
        "description": "Predict whether income exceeds $50K/yr based on census data.",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "rows": 32561,
        "features": 15,
        "target": "income",
        "popularity": 0.95,
        "license": "CC-BY-4.0",
        "file_size_mb": 3.8,
        "columns": ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    }
]

class UCIAdapter(RepositoryAdapter):
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for d in UCI_DATASETS:
            if any(k in query_lower for k in d["keywords"]) or query_lower in d["name"].lower():
                results.append({
                    "name": d["name"],
                    "description": d["description"],
                    "source": "uci",
                    "rows": d["rows"],
                    "features": d["features"],
                    "target": d["target"],
                    "download_url": d["download_url"],
                    "popularity": d["popularity"],
                    "quality_score": 0.95,
                    "license": d["license"],
                    "file_size_mb": d["file_size_mb"]
                })
        return results[:limit]

    def download(self, download_url: str, output_path: str) -> bool:
        try:
            d_info = next((d for d in UCI_DATASETS if d["download_url"] == download_url), None)
            
            req = urllib.request.Request(download_url, headers={'User-Agent': 'ModelSmith-AI-Agent'})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                
            if d_info and d_info.get("columns") is not None:
                text = content.decode('utf-8', errors='ignore')
                import io
                df = pd.read_csv(io.StringIO(text), names=d_info["columns"], header=None, skipinitialspace=True)
                df.to_csv(output_path, index=False)
            else:
                with open(output_path, 'wb') as f:
                    f.write(content)
            return True
        except Exception as e:
            logger.error(f"UCI download error: {e}")
            return False
