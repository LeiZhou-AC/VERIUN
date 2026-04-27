"""Configuration loading helpers."""

import ast
from pathlib import Path

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


def _parse_scalar(value: str):
    """
    Parse a scalar string into Python primitive types.

    Args:
        value: Raw scalar string from config line.

    Returns:
        Parsed Python value.
    """
    raw = value.strip()
    if raw == "":
        return ""

    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _fallback_yaml_like_parser(text: str):
    """
    Parse a minimal YAML-like key-value file without external dependencies.

    Supported syntax:
    - `key: value`
    - `"quoted_key": value`
    - blank lines and `#` comments

    Args:
        text: Raw file contents.

    Returns:
        Parsed dictionary.
    """
    result = {}
    for line in text.splitlines():
        striped = line.strip()
        if not striped or striped.startswith("#"):
            continue
        if ":" not in striped:
            continue
        key_raw, value_raw = striped.split(":", 1)
        key = key_raw.strip().strip("\"'")
        result[key] = _parse_scalar(value_raw)
    return result


def load_config(path: str):
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(text)
        return data or {}

    return _fallback_yaml_like_parser(text)
