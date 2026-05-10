"""Apply the warmup_epochs-as-int patch to src/methods/asyco_divmix.py.

This adds support for specifying warmup_epochs as a single integer in the
config, while keeping backward compatibility with the existing
warmup_epochs_pct + warmup_epochs_floor combination.

Usage:
    cd ~/projects/Bachelor-Project
    python optuna_final_patch/apply_asyco_patch.py

Run this once. Idempotent (safe to re-run; will detect already-patched state).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


TARGET = Path("src/methods/asyco_divmix.py")

# This is the exact block we expect to find in the unmodified file.
# It consumes warmup_epochs_pct and warmup_epochs_floor at __init__ time and
# computes self.warmup_epochs at build() time using _compute_warmup_epochs.
#
# The new version: at __init__, store the raw config keys verbatim.
# At build(), check if a direct `warmup_epochs` int was given; if so, use it.
# Otherwise fall back to the original pct + floor formula.

# --- INIT BLOCK ----------------------------------------------------------
INIT_OLD = (
    '        self.warmup_pct = float(m["warmup_epochs_pct"])\n'
    '        self.warmup_floor = int(m["warmup_epochs_floor"])\n'
)

INIT_NEW = (
    '        # Warmup specification — supports two modes:\n'
    '        # (a) Direct: cfg["method"]["warmup_epochs"] = <int>\n'
    '        # (b) Legacy: cfg["method"]["warmup_epochs_pct"] + ["warmup_epochs_floor"]\n'
    '        # Mode (a) takes precedence if present.\n'
    '        self._warmup_epochs_direct = m.get("warmup_epochs")\n'
    '        if self._warmup_epochs_direct is None:\n'
    '            self.warmup_pct = float(m["warmup_epochs_pct"])\n'
    '            self.warmup_floor = int(m["warmup_epochs_floor"])\n'
    '        else:\n'
    '            self.warmup_pct = None\n'
    '            self.warmup_floor = None\n'
)

# --- BUILD BLOCK ---------------------------------------------------------
# Original computation in build():
BUILD_OLD = (
    '        self.warmup_epochs = _compute_warmup_epochs(\n'
    '            total_epochs=total_epochs,\n'
    '            pct=self.warmup_pct,\n'
    '            floor=self.warmup_floor,\n'
    '        )\n'
)

BUILD_NEW = (
    '        if self._warmup_epochs_direct is not None:\n'
    '            self.warmup_epochs = int(self._warmup_epochs_direct)\n'
    '        else:\n'
    '            self.warmup_epochs = _compute_warmup_epochs(\n'
    '                total_epochs=total_epochs,\n'
    '                pct=self.warmup_pct,\n'
    '                floor=self.warmup_floor,\n'
    '            )\n'
)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} does not exist (run from repo root).", file=sys.stderr)
        return 1

    src = TARGET.read_text()

    # Idempotency check
    if "_warmup_epochs_direct" in src:
        print(f"Already patched (found _warmup_epochs_direct in {TARGET}). No-op.")
        return 0

    # Apply the two replacements
    n_init = src.count(INIT_OLD)
    n_build = src.count(BUILD_OLD)

    if n_init != 1:
        print(
            f"ERROR: expected to find exactly 1 instance of the __init__ block, "
            f"found {n_init}. The file may have diverged from the expected shape.",
            file=sys.stderr,
        )
        print("Expected block:", file=sys.stderr)
        print(INIT_OLD, file=sys.stderr)
        return 2
    if n_build != 1:
        print(
            f"ERROR: expected to find exactly 1 instance of the build() block, "
            f"found {n_build}. The file may have diverged from the expected shape.",
            file=sys.stderr,
        )
        print("Expected block:", file=sys.stderr)
        print(BUILD_OLD, file=sys.stderr)
        return 2

    new = src.replace(INIT_OLD, INIT_NEW, 1).replace(BUILD_OLD, BUILD_NEW, 1)

    backup = TARGET.with_suffix(".py.bak")
    backup.write_text(src)
    TARGET.write_text(new)

    print(f"Patched {TARGET}")
    print(f"Backup saved to {backup}")
    print()
    print("To verify, run:")
    print(f"    python -m py_compile {TARGET}")
    print(f"    diff {backup} {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
