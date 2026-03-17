#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -W 05:00

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m src.utils.merge_fold_probs