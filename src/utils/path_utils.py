"""Path utility functions."""

from pathlib import Path
from typing import Optional


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        path: Path to directory.
    
    Returns:
        Path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get project root directory.
    
    Returns:
        Path to project root (where this file is located).
    """
    # Assuming src/utils/ is at project_root/src/utils/
    return Path(__file__).parent.parent.parent


def get_data_dir(data_type: str = "raw") -> Path:
    """Get data directory path.
    
    Args:
        data_type: Type of data ("raw", "processed", "interim", "external").
    
    Returns:
        Path to data directory.
    """
    root = get_project_root()
    return root / "data" / data_type

