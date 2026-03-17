#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 1:00

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p data/processed/HAM10000/fold_probs
python -m src.utils.collect_fold_probs --fold $FOLD