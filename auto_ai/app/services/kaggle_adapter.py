import urllib.request
import urllib.parse
import json
import os
import base64
import logging
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Any
from auto_ai.app.services.repository_adapter import RepositoryAdapter
from auto_ai.app.infra.db import get_setting

logger = logging.getLogger(__name__)

class KaggleAdapter(RepositoryAdapter):
    def _get_credentials(self) -> tuple:
        # 1. Check settings table
        username = get_setting("kaggle_username")
        key = get_setting("kaggle_key")
        
        # 2. Check environment variables
        if not username or not key:
            username = os.getenv("KAGGLE_USERNAME", "")
            key = os.getenv("KAGGLE_KEY", "")
            
        # 3. Check ~/.kaggle/kaggle.json
        if not username or not key:
            home = Path.home()
            kaggle_json = home / ".kaggle" / "kaggle.json"
            if kaggle_json.exists():
                try:
                    with open(kaggle_json, 'r') as f:
                        creds = json.load(f)
                        username = creds.get("username", "")
                        key = creds.get("key", "")
                except Exception as ex:
                    logger.warning(f"Failed to read kaggle.json: {ex}")
                    
        # 4. Check ~/.kaggle/access_token (as custom backup key)
        if not username or not key:
            access_token_file = Path.home() / ".kaggle" / "access_token"
            if access_token_file.exists():
                try:
                    token = access_token_file.read_text().strip()
                    # If we only have key, we can try a placeholder username or kaggle user
                    username = "kaggle_user"
                    key = token
                except Exception as ex:
                    logger.warning(f"Failed to read access_token: {ex}")
                    
        return username, key

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        username, key = self._get_credentials()
        if not username or not key:
            logger.warning("Kaggle credentials not configured. Skipping Kaggle search.")
            return []
            
        search_term = urllib.parse.quote(query)
        url = f"https://www.kaggle.com/api/v1/datasets/list?search={search_term}"
        logger.info(f"Querying Kaggle: {url}")
        
        try:
            auth_str = f"{username}:{key}"
            encoded_auth = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'ModelSmith-AI-Agent',
                'Authorization': f"Basic {encoded_auth}"
            })
            
            with urllib.request.urlopen(req, timeout=15) as response:
                datasets = json.loads(response.read().decode('utf-8'))
                
            results = []
            for d in datasets[:limit]:
                ref = d.get("ref", "")
                title = d.get("title", "")
                desc = d.get("description", "") or d.get("subtitle", "") or title
                download_count = d.get("downloadCount", 0)
                
                # Kaggle datasets don't expose target easily, we set default None to let Ranker score it
                results.append({
                    "name": title,
                    "description": desc,
                    "source": "kaggle",
                    "rows": None,
                    "features": None,
                    "target": None,
                    # We store ref as download URL so we can construct it on download phase
                    "download_url": ref,
                    "popularity": min(1.0, download_count / 10000.0),
                    "quality_score": 0.8,
                    "license": d.get("licenseName", "Unknown"),
                    "file_size_mb": 10.0
                })
            return results
        except Exception as e:
            logger.error(f"Kaggle search error: {e}")
            return []

    def download(self, download_url: str, output_path: str) -> bool:
        username, key = self._get_credentials()
        if not username or not key:
            return False
            
        # download_url contains owner/dataset-name
        url = f"https://www.kaggle.com/api/v1/datasets/download/{download_url}"
        logger.info(f"Downloading from Kaggle: {url}")
        
        try:
            auth_str = f"{username}:{key}"
            encoded_auth = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'ModelSmith-AI-Agent',
                'Authorization': f"Basic {encoded_auth}"
            })
            
            with urllib.request.urlopen(req, timeout=45) as response:
                content = response.read()
                
            # Kaggle download is a ZIP archive
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # Find first CSV file
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    logger.error("No CSV files found in Kaggle ZIP archive")
                    return False
                
                # Extract first CSV file to output path
                first_csv = csv_files[0]
                logger.info(f"Extracting {first_csv} from Kaggle ZIP archive")
                with open(output_path, 'wb') as f_out:
                    f_out.write(z.read(first_csv))
                    
            return True
        except Exception as e:
            logger.error(f"Kaggle download error: {e}")
            return False
