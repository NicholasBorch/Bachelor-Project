#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 00:15
#BSUB -J balanced_idn_cv[1-10]
#BSUB -o runs/hpc/balanced_noise/logs/idn_fold_%I.out
#BSUB -e runs/hpc/balanced_noise/logs/idn_fold_%I.err

# Creates BOTH cv_balanced_standard/ and cv_balanced_normalized/ in a single pass.
# Standard IDN is reference-only and not used in training experiments.
# No GPU needed — noise injection is CPU-only.

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

FOLD=$(( LSB_JOBINDEX - 1 ))

echo "Creating balanced standard + normalized IDN folds — fold ${FOLD}"
python -m src.utils.create_balanced_cv_folds --fold "${FOLD}"
