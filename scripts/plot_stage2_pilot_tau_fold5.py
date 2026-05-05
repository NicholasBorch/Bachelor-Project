import json
from pathlib import Path

import matplotlib.pyplot as plt


root = Path("results/pilot_stage2_tau_fold5_configs_0")
out = root / "plots"
out.mkdir(parents=True, exist_ok=True)

datasets = ["balanced", "imbalanced"]
methods = ["elr", "asyco_divmix"]
optims = ["sgd", "adam"]
models = ["resnet34_pretrained", "resnet34_scratch"]
taus = ["tau_00", "tau_20"]
fold = "fold_05"


def load_curve(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    epochs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]
    val_ba = [r["val_balanced_accuracy"] for r in rows]

    return epochs, val_ba


for dataset in datasets:
    for method in methods:
        for optim in optims:
            for model in models:
                curves = {}

                for tau in taus:
                    log_path = (
                        root / tau / dataset / method / optim / model
                        / fold / "training_log.jsonl"
                    )

                    if not log_path.exists():
                        print("Missing:", log_path)
                        continue

                    curves[tau] = load_curve(log_path)

                if not curves:
                    continue

                plt.figure(figsize=(8, 5))

                for tau, (epochs, val_ba) in curves.items():
                    best_epoch = epochs[val_ba.index(max(val_ba))]
                    best_val = max(val_ba)

                    plt.plot(
                        epochs,
                        val_ba,
                        linewidth=1.4,
                        label=f"{tau} (best={best_epoch}, max={best_val:.3f})",
                    )
                    plt.axvline(best_epoch, linestyle="--", alpha=0.35)

                plt.xlabel("epoch")
                plt.ylabel("clean validation balanced accuracy")
                plt.title(f"{dataset} / {method} / {optim} / {model} / fold_05")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()

                out_path = (
                    out
                    / f"{dataset}_{method}_{optim}_{model}_fold05_tau00_vs_tau20.png"
                )
                plt.savefig(out_path, dpi=150)
                plt.close()

                print("Saved:", out_path)

print("Done.")