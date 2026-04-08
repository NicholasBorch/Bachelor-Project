#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 00:20
#BSUB -J balanced_fold_probs[1-10]
#BSUB -o runs/hpc/balanced_noise/logs/fold_probs_%I.out
#BSUB -e runs/hpc/balanced_noise/logs/fold_probs_%I.err

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

FOLD=$(( LSB_JOBINDEX - 1 ))

echo "Collecting balanced OOF probs — fold ${FOLD}"
python -m src.utils.collect_balanced_fold_probs --fold "${FOLD}"
