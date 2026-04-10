#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 00:10
#BSUB -J balanced_fd_cv[1-10]
#BSUB -o runs/hpc/balanced_noise/logs/fd_fold_%I.out
#BSUB -e runs/hpc/balanced_noise/logs/fd_fold_%I.err

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

FOLD=$(( LSB_JOBINDEX - 1 ))

echo "Creating balanced feature-driven IDN folds — fold ${FOLD}"
python -m src.utils.prepare_balanced_cv_feature_driven --fold "${FOLD}"
