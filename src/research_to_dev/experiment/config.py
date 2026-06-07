"""YAML config loading helper for experiment setup.

``load_config_yaml()`` reads an optional ``.research-to-dev/config.yaml``
and returns its parsed contents.  If the file doesn't exist, an empty dict
is returned — config.yaml is always optional (D6).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config_yaml(path: str) -> dict:
    """Load a YAML config file, returning an empty dict if it doesn't exist.

    Args:
        path: Filesystem path to the YAML file (e.g.
            ``.research-to-dev/config.yaml``).

    Returns:
        Parsed YAML content as a dict.  Empty dict if the file is missing.

    Raises:
        yaml.YAMLError: If the file exists but cannot be parsed.
    """
    config_file = Path(path)
    if not config_file.exists():
        return {}

    raw = config_file.read_text(encoding="utf-8")
    return yaml.safe_load(raw) or {}
