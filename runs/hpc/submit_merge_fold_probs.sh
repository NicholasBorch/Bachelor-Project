#!/bin/bash
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -W 0:10

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m src.utils.merge_fold_probs