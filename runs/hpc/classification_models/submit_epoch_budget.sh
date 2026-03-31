#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 02:00

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m src.utils.find_epoch_budget \
    --fold $FOLD \
    --epochs 100 \
    --val_frac 0.15