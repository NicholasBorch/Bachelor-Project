import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path("results/pilot_stage2_tau20_fold03")
out = Path("results/pilot_stage2_tau20_fold03/plots")
out.mkdir(parents=True, exist_ok=True)

datasets = ["balanced", "imbalanced"]
methods = ["elr", "asyco_divmix"]
optims = ["sgd", "adam"]
models = ["resnet34_pretrained", "resnet34_scratch"]
fold = "fold_03"

for dataset in datasets:
    for method in methods:
        for optim in optims:
            for model in models:
                log_path = root / dataset / method / optim / model / fold / "training_log.jsonl"

                if not log_path.exists():
                    print("Missing:", log_path)
                    continue

                rows = []
                with open(log_path) as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))

                epochs = [r.get("epoch", i + 1) for i, r in enumerate(rows)]
                val_ba = [r["val_balanced_accuracy"] for r in rows]

                plt.figure(figsize=(7, 4))
                plt.plot(epochs, val_ba, linewidth=1)

                best_epoch = epochs[val_ba.index(max(val_ba))]
                plt.axvline(best_epoch, linestyle="--")

                plt.xlabel("epoch")
                plt.ylabel("val balanced accuracy")
                plt.title(f"{dataset} / {method} / {optim} / {model} (best={best_epoch})")
                plt.grid(alpha=0.3)
                plt.tight_layout()

                out_path = out / f"{dataset}_{method}_{optim}_{model}_fold{fold}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()

                print("Saved:", out_path)

print("Done.")