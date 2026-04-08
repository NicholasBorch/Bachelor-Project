#!/bin/bash
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 00:10
#BSUB -J balanced_merge_probs
#BSUB -o runs/hpc/balanced_noise/logs/merge_probs.out
#BSUB -e runs/hpc/balanced_noise/logs/merge_probs.err

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Merging balanced OOF fold probabilities..."
python -m src.utils.merge_balanced_fold_probs
