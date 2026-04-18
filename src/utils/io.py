"""Path resolution and YAML config loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Resolve the project root directory by walking up from this file's location.

    This file lives at <root>/src/utils/io.py, so root is parents[2].
    """
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(
    *relative_paths: str,
    configs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge multiple YAML configs from the configs/ directory.

    Configs are namespaced by their parent directory:
      - base.yaml                -> merged at top level
      - data/<name>.yaml         -> merged under cfg["data"]
      - method/<name>.yaml       -> merged under cfg["method"]
      - optim/<name>.yaml        -> merged under cfg["optim"]
      - model/<name>.yaml        -> merged under cfg["model"]
      - noise/<name>.yaml        -> merged under cfg["noise"]

    This namespacing prevents collisions between, e.g., `name` fields in
    different config sections.

    Example:
        cfg = load_config("base.yaml", "data/imbalanced.yaml", "method/elr.yaml")
        cfg["seed"]            # from base.yaml
        cfg["data"]["name"]    # "imbalanced"
        cfg["method"]["name"]  # "elr"
    """
    if configs_dir is None:
        configs_dir = project_root() / "configs"
    configs_dir = Path(configs_dir)

    merged: dict[str, Any] = {}
    for rel in relative_paths:
        data = load_yaml(configs_dir / rel)
        if data is None:
            continue
        parts = Path(rel).parts
        if len(parts) == 1:
            # Top-level file (e.g. base.yaml): merge directly
            merged.update(data)
        else:
            # Namespaced by first directory component
            namespace = parts[0]
            if namespace not in merged:
                merged[namespace] = {}
            merged[namespace].update(data)
    return merged


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Dump a dict to YAML, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist, return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
