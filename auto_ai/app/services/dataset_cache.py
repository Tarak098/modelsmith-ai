import os
import logging
import shutil
from pathlib import Path
from auto_ai.app.infra.db import get_setting, get_setting_int

logger = logging.getLogger(__name__)

class DatasetCache:
    @staticmethod
    def get_cache_dir() -> Path:
        cache_dir_str = get_setting("cache_dir") or "data/cache"
        from auto_ai.app.config import settings
        cache_path = Path(cache_dir_str)
        if not cache_path.is_absolute():
            cache_path = settings.DATA_DIR / cache_dir_str
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @classmethod
    def check(cls, name: str) -> str:
        cache_dir = cls.get_cache_dir()
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()
        if not safe_name.endswith('.csv'):
            safe_name += '.csv'
            
        file_path = cache_dir / safe_name
        if file_path.exists():
            logger.info(f"Cache hit: {file_path}")
            return str(file_path)
        return None

    @classmethod
    def add(cls, name: str, source_path: str) -> str:
        cache_dir = cls.get_cache_dir()
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()
        if not safe_name.endswith('.csv'):
            safe_name += '.csv'
            
        dest_path = cache_dir / safe_name
        
        if os.path.abspath(source_path) != os.path.abspath(dest_path):
            shutil.copy2(source_path, dest_path)
            
        logger.info(f"Added to cache: {dest_path}")
        cls.cleanup()
        return str(dest_path)

    @classmethod
    def cleanup(cls):
        cache_dir = cls.get_cache_dir()
        max_size_mb = get_setting_int("max_cache_size_mb", 500)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        files = []
        total_size = 0
        for entry in os.scandir(cache_dir):
            if entry.is_file():
                stat = entry.stat()
                files.append((entry.path, stat.st_mtime, stat.st_size))
                total_size += stat.st_size
                
        if total_size > max_size_bytes:
            logger.info(f"Cache size ({total_size / (1024*1024):.1f} MB) exceeds limit ({max_size_mb} MB). Cleaning oldest files.")
            files.sort(key=lambda x: x[1])
            
            for path, _, size in files:
                try:
                    os.remove(path)
                    logger.info(f"Removed cache file: {path}")
                    total_size -= size
                    if total_size <= max_size_bytes:
                        break
                except Exception as ex:
                    logger.error(f"Failed to remove cache file {path}: {ex}")
