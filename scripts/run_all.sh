#!/bin/bash
# scripts/run_all.sh
#
# Local, single-machine convenience wrapper that runs Stages 0-4 in sequence
# with sensible defaults.
#
# ╔════════════════════════════════════════════════════════════════════════╗
# ║  THIS IS NOT THE RECOMMENDED WORKFLOW.                                 ║
# ║                                                                        ║
# ║  Stage 3 alone is ~3,400 GPU-hours; even on a fast single GPU that is  ║
# ║  weeks to months of wall-clock time. For the real campaign, submit     ║
# ║  jobs to the cluster via hpc/submit_stage*.sh at 10-12 parallel jobs.  ║
# ║                                                                        ║
# ║  Use this script for:                                                  ║
# ║      - End-to-end smoke tests on a tiny subset.                        ║
# ║      - Reproducibility checks after code edits.                        ║
# ║      - Off-HPC debugging where you want to step through the stages.    ║
# ╚════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash scripts/run_all.sh                 # run every stage with defaults
#   bash scripts/run_all.sh --skip-stage3   # everything except the expensive bit
#   bash scripts/run_all.sh --smoke         # subset grid for a quick smoke test
#
# Preconditions:
#   - HAM10000 has been downloaded into data/raw (see README).
#   - Python env has the requirements.txt packages installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SKIP_STAGE3=0
SMOKE=0
for arg in "$@"; do
    case "${arg}" in
        --skip-stage3) SKIP_STAGE3=1 ;;
        --smoke)       SMOKE=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown flag: ${arg}" >&2; exit 2 ;;
    esac
done

banner() {
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
}

# Method list — keep in sync with src/methods/__init__.py and
# scripts/stage3_train.py.
METHODS="baseline sce elr asyco asyco_divmix"

banner "Stage 0 — prepare dataset"
python -m scripts.stage0_prepare_dataset

banner "Stage 1a — create fold assignments"
for dataset in balanced imbalanced; do
    python -m scripts.stage1a_create_folds --dataset "${dataset}"
done

banner "Stage 1b — collect OOF probabilities (requires GPU)"
for dataset in balanced imbalanced; do
    for fold in $(seq 0 9); do
        python -m scripts.stage1b_collect_oof_probs --dataset "${dataset}" --fold "${fold}"
    done
    python -m scripts.stage1b_merge_oof_probs --dataset "${dataset}"
done

banner "Stage 1c — inject noise"
for dataset in balanced imbalanced; do
    for noise_type in standard normalized feature_driven; do
        for fold in $(seq 0 9); do
            python -m scripts.stage1c_inject_noise \
                --dataset "${dataset}" --noise-type "${noise_type}" --fold "${fold}"
        done
    done
done

banner "Stage 1d — characterize noise"
python -m scripts.stage1d_characterize_noise

banner "Stage 1e — human comparison"
python -m scripts.stage1e_human_comparison

banner "Stage 2 — select epoch budgets (requires GPU)"
for dataset in balanced imbalanced; do
    for method in ${METHODS}; do
        for fold in $(seq 0 9); do
            python -m scripts.stage2_select_epoch_budget \
                --dataset "${dataset}" --method "${method}" --fold "${fold}"
        done
        python -m scripts.stage2_aggregate_epoch_budget \
            --dataset "${dataset}" --method "${method}"
    done
done

if [[ "${SKIP_STAGE3}" -eq 1 ]]; then
    echo
    echo "Skipping Stage 3 per --skip-stage3. Run:"
    echo "    bash hpc/submit_stage3.sh"
    echo "on the cluster to complete the campaign."
else
    banner "Stage 3 — main training (THIS IS THE EXPENSIVE ONE)"
    if [[ "${SMOKE}" -eq 1 ]]; then
        echo "Smoke mode: running only (baseline, balanced, pretrained, sgd, tau=0.3, fold=0)."
        python -m scripts.stage3_train \
            --method baseline --dataset balanced \
            --init pretrained --optim sgd --tau 0.3 --fold 0
    else
        echo "Full grid: 2,400 jobs single-threaded. You will be waiting a long time."
        echo "Ctrl-C now and use hpc/submit_stage3.sh instead unless you really mean this."
        read -rp "Proceed anyway? [y/N] " reply
        if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
            echo "Aborted at Stage 3. Use hpc/submit_stage3.sh on the cluster."
            exit 0
        fi
        python -m hpc.generate_stage3_jobs | \
            sed -E 's/^bsub [^"]+"([^"]+)".*/\1/' | \
            while read -r cmd; do eval "${cmd}"; done
    fi
fi

banner "Stage 4 — analysis"
python -m scripts.stage4_analyze --force

echo
echo "All stages complete. Figures and tables are in results/final_figures/."
