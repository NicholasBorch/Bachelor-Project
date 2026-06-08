"""Per-stage manifest files, written on stage completion for prerequisite checks and cheap inspection."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    """Return the current git commit short-hash, or None if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_manifest(
    path: str | Path,
    stage: str,
    params: dict[str, Any],
    outputs: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a manifest JSON describing what a stage produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "params": params,
        "outputs": outputs or [],
    }
    if extra:
        manifest.update(extra)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest file."""
    with open(path, "r") as f:
        return json.load(f)