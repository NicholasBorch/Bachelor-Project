# Shared I/O and label utilities used across classification and segmentation.

from pathlib import Path
from typing import Dict, List, Tuple


def class_mapping(classes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Stable bidirectional mapping between string labels and integer indices
    classes_sorted = sorted(list(set(classes)))
    c2i = {c: i for i, c in enumerate(classes_sorted)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c

def project_root() -> Path:
    # Resolves the repository root from any file two levels below src/
    return Path(__file__).resolve().parents[2]
