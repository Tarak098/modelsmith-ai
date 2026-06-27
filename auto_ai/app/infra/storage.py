import os
import shutil
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple, Any
from auto_ai.app.config import settings
from auto_ai.app.utils.logging import logger
from auto_ai.app.utils.exceptions import ModelSmithException

class StorageManager:
    @staticmethod
    def get_run_dir(project_id: str) -> Path:
        """Returns the base directory for a specific project run, creating it if it doesn't exist."""
        run_dir = settings.DATA_DIR / "runs" / project_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create subfolders inside the run directory
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        return run_dir

    @staticmethod
    def save_dataset(project_id: str, df: pd.DataFrame, filename: str) -> Path:
        """Save a pandas DataFrame as a CSV file in the run directory."""
        run_dir = StorageManager.get_run_dir(project_id)
        filepath = run_dir / filename
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved dataset to {filepath}")
            return filepath
        except Exception as e:
            raise ModelSmithException(f"Failed to save dataset {filename} for project {project_id}: {e}")

    @staticmethod
    def load_dataset(project_id: str, filename: str) -> pd.DataFrame:
        """Load a CSV dataset from the run directory as a pandas DataFrame."""
        run_dir = StorageManager.get_run_dir(project_id)
        filepath = run_dir / filename
        if not filepath.exists():
            raise ModelSmithException(f"Dataset {filename} does not exist for project {project_id} at {filepath}")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise ModelSmithException(f"Failed to load dataset {filename} for project {project_id}: {e}")

    @staticmethod
    def save_model(project_id: str, model: Any, model_name: str) -> Path:
        """Serialize and save a trained model file using joblib."""
        run_dir = StorageManager.get_run_dir(project_id)
        filepath = run_dir / "models" / f"{model_name}.joblib"
        try:
            joblib.dump(model, filepath)
            logger.info(f"Saved model to {filepath}")
            return filepath
        except Exception as e:
            raise ModelSmithException(f"Failed to save model {model_name} for project {project_id}: {e}")

    @staticmethod
    def load_model(project_id: str, model_name: str) -> Any:
        """Load a serialized model using joblib."""
        run_dir = StorageManager.get_run_dir(project_id)
        filepath = run_dir / "models" / f"{model_name}.joblib"
        if not filepath.exists():
            raise ModelSmithException(f"Model file {model_name} does not exist at {filepath}")
        try:
            return joblib.load(filepath)
        except Exception as e:
            raise ModelSmithException(f"Failed to load model {model_name}: {e}")

    @staticmethod
    def get_plot_path(project_id: str, plot_name: str) -> Path:
        """Get the destination path to save a Matplotlib plot image."""
        run_dir = StorageManager.get_run_dir(project_id)
        return run_dir / "plots" / f"{plot_name}.png"

    @staticmethod
    def save_uploaded_file(uploaded_file_bytes: bytes, filename: str) -> Path:
        """Save a user-uploaded file into the upload repository, returning its path."""
        upload_dir = settings.DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        filepath = upload_dir / filename
        try:
            with open(filepath, "wb") as f:
                f.write(uploaded_file_bytes)
            logger.info(f"Successfully saved uploaded file to {filepath}")
            return filepath
        except Exception as e:
            raise ModelSmithException(f"Failed to save uploaded file {filename}: {e}")
            
    @staticmethod
    def copy_file_to_run(project_id: str, source_path: Path, dest_filename: str) -> Path:
        """Copy a file from outside the run folder (like an upload) into the run folder."""
        run_dir = StorageManager.get_run_dir(project_id)
        dest_path = run_dir / dest_filename
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {source_path} to run folder as {dest_path}")
            return dest_path
        except Exception as e:
            raise ModelSmithException(f"Failed to copy file to run folder: {e}")
            
    @staticmethod
    def save_report(project_id: str, report_content: str, filename: str) -> Path:
        """Save a generated markdown or HTML report to the run folder."""
        run_dir = StorageManager.get_run_dir(project_id)
        filepath = run_dir / "reports" / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Saved report to {filepath}")
            return filepath
        except Exception as e:
            raise ModelSmithException(f"Failed to save report: {e}")
