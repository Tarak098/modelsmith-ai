import os
import re
from pathlib import Path
from auto_ai.app.config import settings
from auto_ai.app.utils.exceptions import ModelSmithException

def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from a filename.
    """
    # Keep only alphanumeric, hyphens, underscores, and dots
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '', filename)
    # Remove leading dots or slashes to prevent traversal
    sanitized = sanitized.lstrip('./\\')
    return sanitized if sanitized else "uploaded_dataset.csv"

def validate_secure_path(path: Path) -> Path:
    """
    Verify that the target path remains entirely inside the project data directory
    to prevent path traversal attacks.
    """
    resolved_path = path.resolve()
    resolved_data_dir = settings.DATA_DIR.resolve()
    
    if not str(resolved_path).startswith(str(resolved_data_dir)):
        raise ModelSmithException(
            "Security Violation: Path traversal attempt detected.",
            {"violating_path": str(resolved_path)}
        )
    return resolved_path

def scan_prompt_injection(prompt: str) -> str:
    """
    Scans the input user prompt for basic prompt injection and system control command patterns.
    Returns the sanitized prompt or raises a ModelSmithException if highly unsafe.
    """
    unsafe_patterns = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)system\s+command",
        r"(?i)rm\s+-rf",
        r"(?i)drop\s+database",
        r"(?i)delete\s+files",
        r"<\s*script\s*>"
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, prompt):
            raise ModelSmithException(
                "Unsafe input detected: The instruction contains potentially malicious system override statements.",
                {"violating_pattern": pattern}
            )
            
    # Simple HTML sanitization
    clean_prompt = re.sub(r'<[^>]*>', '', prompt)
    return clean_prompt.strip()
