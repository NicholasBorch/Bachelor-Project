#!/bin/bash
# Manual single-fold resubmission for normalised IDN.
# Usage: bash runs/hpc/submit_normalized_cv.sh <fold>

set -euo pipefail
cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs
export PYTHONUNBUFFERED=1

FOLD=${1:?Usage: bash submit_normalized_cv.sh <fold>}

bsub \
    -J "cvnorm${FOLD}" \
    -q gpuv100 \
    -n 8 \
    -R "span[hosts=1]" \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 0:30 \
    -oo logs/cvnorm_${FOLD}.out \
    -eo logs/cvnorm_${FOLD}.err \
    python -m src.utils.prepare_classification_cv --fold $FOLD --method normalized