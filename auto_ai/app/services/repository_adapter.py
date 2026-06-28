from abc import ABC, abstractmethod
from typing import List, Dict, Any

class RepositoryAdapter(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the repository and return a list of standardized datasets.
        Return schema:
        {
            "name": str,
            "description": str,
            "source": str,
            "rows": int,
            "features": int,
            "target": str,
            "download_url": str,
            "popularity": float,      # 0.0 to 1.0 scale
            "quality_score": float,    # 0.0 to 1.0 scale
            "license": str,
            "file_size_mb": float
        }
        """
        pass

    @abstractmethod
    def download(self, download_url: str, output_path: str) -> bool:
        """
        Download the dataset to a local path.
        Returns True if successful, False otherwise.
        """
        pass
