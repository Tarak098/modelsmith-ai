import urllib.request
import urllib.parse
import json
import logging
from typing import List, Dict, Any
from auto_ai.app.services.repository_adapter import RepositoryAdapter

logger = logging.getLogger(__name__)

class OpenMLAdapter(RepositoryAdapter):
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        search_term = urllib.parse.quote(query)
        url = f"https://www.openml.org/api/v1/json/data/list/search_term/{search_term}"
        logger.info(f"Querying OpenML: {url}")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ModelSmith-AI-Agent'})
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
            datasets = res_data.get("data", {}).get("dataset", [])
            if not isinstance(datasets, list):
                datasets = [datasets]
                
            results = []
            for d in datasets[:limit]:
                did = d.get("did")
                name = d.get("name", "")
                
                # Fetch detailed metadata
                detail_url = f"https://www.openml.org/api/v1/json/data/{did}"
                try:
                    req_detail = urllib.request.Request(detail_url, headers={'User-Agent': 'ModelSmith-AI-Agent'})
                    with urllib.request.urlopen(req_detail, timeout=10) as detail_resp:
                        d_detail = json.loads(detail_resp.read().decode('utf-8')).get("data_set_description", {})
                except Exception as ex:
                    logger.warning(f"Failed to fetch details for OpenML did={did}: {ex}")
                    d_detail = {}
                
                desc = d_detail.get("description", "")
                download_url = d_detail.get("url", "")
                
                qualities = d_detail.get("qualities", [])
                rows = None
                features = None
                target = d_detail.get("default_target_attribute", "")
                
                if isinstance(qualities, list):
                    for q in qualities:
                        name_q = q.get("name", "")
                        value_q = q.get("value", "")
                        if name_q == "NumberOfInstances":
                            try: rows = int(float(value_q))
                            except: pass
                        elif name_q == "NumberOfFeatures":
                            try: features = int(float(value_q))
                            except: pass
                
                results.append({
                    "name": name,
                    "description": desc,
                    "source": "openml",
                    "rows": rows,
                    "features": features,
                    "target": target,
                    "download_url": download_url,
                    "popularity": 0.8,
                    "quality_score": 0.9 if rows and features else 0.5,
                    "license": d_detail.get("licence", "CC-BY"),
                    "file_size_mb": 5.0
                })
            return results
        except Exception as e:
            logger.error(f"OpenML search error: {e}")
            return []

    def download(self, download_url: str, output_path: str) -> bool:
        if not download_url:
            return False
        try:
            req = urllib.request.Request(download_url, headers={'User-Agent': 'ModelSmith-AI-Agent'})
            with urllib.request.urlopen(req, timeout=30) as response:
                content_type = response.headers.get('Content-Type', '')
                content = response.read()
                
            if download_url.endswith('.arff') or 'arff' in content_type:
                logger.info("Converting OpenML ARFF format to CSV")
                arff_text = content.decode('utf-8', errors='ignore')
                df = arff_to_dataframe(arff_text)
                df.to_csv(output_path, index=False)
            else:
                with open(output_path, 'wb') as f:
                    f.write(content)
            return True
        except Exception as e:
            logger.error(f"OpenML download error: {e}")
            return False

def arff_to_dataframe(arff_text: str):
    import pandas as pd
    import io
    lines = arff_text.splitlines()
    columns = []
    data_started = False
    csv_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line.lower().startswith('@attribute'):
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[1].strip("'\"")
                columns.append(col_name)
        elif line.lower().startswith('@data'):
            data_started = True
            continue
        elif data_started:
            processed_line = line.replace('?', '')
            csv_lines.append(processed_line)
            
    csv_data = "\n".join(csv_lines)
    df = pd.read_csv(io.StringIO(csv_data), names=columns, header=None)
    return df
