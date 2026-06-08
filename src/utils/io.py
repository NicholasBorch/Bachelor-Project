"""Path resolution and YAML config loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Resolve the project root (this file is <root>/src/utils/io.py, so parents[2])."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(
    *relative_paths: str,
    configs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge YAML configs from configs/, namespacing each by its parent dir (data/, method/, optim/, model/, noise/) so section `name` fields don't collide."""
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