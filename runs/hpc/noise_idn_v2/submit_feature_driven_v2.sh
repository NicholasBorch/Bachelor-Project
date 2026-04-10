#!/bin/bash
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 00:30
#BSUB -J fd_v2_cv[1-10]
#BSUB -o runs/hpc/noise_idn_v2/logs/fd_v2_fold_%I.out
#BSUB -e runs/hpc/noise_idn_v2/logs/fd_v2_fold_%I.err

# Generates feature-driven IDN v2 (argmax variant) fold CSVs on the
# full imbalanced dataset (7,470 samples).
# No GPU needed — purely CPU computation on precomputed OOF probs.
# Requires fold_probs/fold_probs_full.npy to exist (from the original pipeline).

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

FOLD=$(( LSB_JOBINDEX - 1 ))

echo "Creating feature-driven IDN v2 (argmax) folds — fold ${FOLD}"
python -m src.utils.prepare_cv_feature_driven_v2 --fold "${FOLD}"
