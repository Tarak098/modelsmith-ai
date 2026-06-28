import pandas as pd
import logging
from sklearn.datasets import (
    fetch_california_housing, load_diabetes, load_iris, load_breast_cancer, load_wine
)
from auto_ai.app.services.repository_adapter import RepositoryAdapter
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

SKLEARN_DATASETS = [
    {
        "name": "California Housing",
        "keywords": ["house", "housing", "california", "real estate", "price", "property"],
        "loader": fetch_california_housing,
        "target_name": "MedHouseVal",
        "description": "California housing dataset containing average house values and demographic features by block group.",
        "rows": 20640,
        "features": 8,
        "type": "regression"
    },
    {
        "name": "Diabetes Dataset",
        "keywords": ["diabetes", "medical", "health", "clinical", "glucose"],
        "loader": load_diabetes,
        "target_name": "target",
        "description": "Diabetes patient progression dataset containing physiological baseline indicators.",
        "rows": 442,
        "features": 10,
        "type": "regression"
    },
    {
        "name": "Iris Flower Classification",
        "keywords": ["iris", "flower", "setosa", "versicolor", "virginica"],
        "loader": load_iris,
        "target_name": "target",
        "description": "Iris flower classification dataset containing sepal and petal measurements.",
        "rows": 150,
        "features": 4,
        "type": "classification"
    },
    {
        "name": "Breast Cancer Diagnostic",
        "keywords": ["cancer", "breast", "malignant", "tumor", "diagnostic"],
        "loader": load_breast_cancer,
        "target_name": "target",
        "description": "Wisconsin breast cancer diagnostic dataset containing cytological features.",
        "rows": 569,
        "features": 30,
        "type": "classification"
    },
    {
        "name": "Wine Recognition",
        "keywords": ["wine", "recognition", "cultivar", "chemical", "alcohol"],
        "loader": load_wine,
        "target_name": "target",
        "description": "Wine recognition dataset containing chemical analysis of wines grown in Italy.",
        "rows": 178,
        "features": 13,
        "type": "classification"
    }
]

class SklearnAdapter(RepositoryAdapter):
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for d in SKLEARN_DATASETS:
            if any(k in query_lower for k in d["keywords"]) or query_lower in d["name"].lower():
                results.append({
                    "name": d["name"],
                    "description": d["description"],
                    "source": "sklearn",
                    "rows": d["rows"],
                    "features": d["features"],
                    "target": d["target_name"],
                    "download_url": d["name"],
                    "popularity": 0.9,
                    "quality_score": 1.0,
                    "license": "BSD",
                    "file_size_mb": 1.0
                })
        return results[:limit]

    def download(self, download_url: str, output_path: str) -> bool:
        try:
            d_info = next((d for d in SKLEARN_DATASETS if d["name"] == download_url), None)
            if not d_info:
                return False
            
            logger.info(f"Loading scikit-learn built-in: {d_info['name']}")
            data = d_info["loader"](as_frame=True)
            df = data.frame
            
            if d_info["target_name"] != "target" and "target" in df.columns:
                df = df.rename(columns={"target": d_info["target_name"]})
                
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Sklearn load error: {e}")
            return False
