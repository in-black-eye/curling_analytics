import os
from pathlib import Path
import yaml

def load_config(config_path: str = "configs/config.yaml"):
    """Load YAML config.
    Resolves relative path against project root so it works from any CWD.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        # Try relative to current working directory first
        if not cfg_path.exists():
            # Fallback: resolve against project root (parent of src)
            project_root = Path(__file__).resolve().parents[1]
            cfg_path = project_root / config_path
    with open(cfg_path, 'r') as file:
        return yaml.safe_load(file)
