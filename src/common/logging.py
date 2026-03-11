# Structured results logging for classification experiments.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EpochRecord:
    # Metrics recorded at the end of one training epoch
    epoch:      int
    train_loss: float


@dataclass
class RunConfig:
    # Full configuration for one training run — written once per run for reproducibility
    method:      str
    tau:         float
    outer_fold:  int
    seed:        int
    backbone:    str
    epochs:      int
    batch_size:  int
    lr:          float
    image_size:  int
    noise_type:  str
    extra:       Optional[Dict] = None


class ResultsLogger:
    def __init__(self, output_dir: Path, config: RunConfig) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config     = config
        self.epoch_log: List[EpochRecord] = []
        self._write_json(asdict(config), "config.json")

    def log_epoch(self, epoch: int, train_loss: float) -> None:
        record = EpochRecord(epoch=epoch, train_loss=train_loss)
        self.epoch_log.append(asdict(record))
        self._write_json(self.epoch_log, "training_log.json")

    def log_test_metrics(self, metrics: Dict) -> None:
        self._write_json(metrics, "test_metrics.json")

    def _write_json(self, obj, filename: str) -> None:
        with open(self.output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


def make_output_dir(
    results_root: Path,
    method: str,
    tau: float,
    outer_fold: int,
    noise_type: str = "standard_idn",
) -> Path:
    tau_tag  = "clean" if tau == 0.0 else f"tau{int(tau * 100):02d}"
    fold_tag = f"fold_{outer_fold:02d}"
    return results_root / method / noise_type / tau_tag / fold_tag


def load_run_results(run_dir: Path) -> Optional[Dict]:
    config_path  = run_dir / "config.json"
    metrics_path = run_dir / "test_metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None
    with open(config_path) as f:
        config = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {"config": config, "metrics": metrics}


def load_all_results(
    results_root: Path,
    method: str,
    noise_type: str = "standard_idn",
) -> List[Dict]:
    method_dir = results_root / method / noise_type
    if not method_dir.exists():
        return []
    runs = []
    for run_dir in sorted(method_dir.glob("*/fold_*")):
        result = load_run_results(run_dir)
        if result is not None:
            runs.append(result)
    return runs