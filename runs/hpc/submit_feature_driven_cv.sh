#!/bin/bash
# Manual single-fold resubmission for feature-driven IDN.
# Usage: bash runs/hpc/submit_feature_driven_cv.sh <fold>

set -euo pipefail
cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs
export PYTHONUNBUFFERED=1

FOLD=${1:?Usage: bash submit_feature_driven_cv.sh <fold>}

bsub \
    -J "cvfd${FOLD}" \
    -q gpuv100 \
    -n 8 \
    -R "span[hosts=1]" \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 0:20 \
    -oo logs/cvfd_${FOLD}.out \
    -eo logs/cvfd_${FOLD}.err \
    python -m src.utils.prepare_classification_cv_feature_driven --fold $FOLD