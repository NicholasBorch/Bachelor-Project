#!/bin/bash
# Manual single-fold resubmission for fold prob collection.
# Usage: bash runs/hpc/submit_fold_probs.sh <fold>

set -euo pipefail
cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs
mkdir -p data/processed/HAM10000/fold_probs
export PYTHONUNBUFFERED=1

FOLD=${1:?Usage: bash submit_fold_probs.sh <fold>}

bsub \
    -J "foldprobs${FOLD}" \
    -q gpuv100 \
    -n 8 \
    -R "span[hosts=1]" \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 1:00 \
    -oo logs/foldprobs_${FOLD}.out \
    -eo logs/foldprobs_${FOLD}.err \
    python -m src.utils.collect_fold_probs --fold $FOLD