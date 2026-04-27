"""Configuration loading helpers."""

from pathlib import Path

import yaml


def load_config(path: str):
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
