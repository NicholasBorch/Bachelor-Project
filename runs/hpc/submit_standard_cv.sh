#!/bin/bash
# Manual single-fold resubmission for standard IDN.
# Usage: bash runs/hpc/submit_standard_cv.sh <fold>

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs
VENV="$HOME/projects/Bachelor-Project/.venv/bin/activate"

FOLD=${1:?Usage: bash submit_standard_cv.sh <fold>}

bsub \
    -J "cvstd${FOLD}" \
    -q gpuv100 \
    -n 8 \
    -R "span[hosts=1]" \
    -R "rusage[mem=16000]" \
    -gpu "num=1" \
    -W 0:30 \
    -oo logs/cvstd_${FOLD}.out \
    -eo logs/cvstd_${FOLD}.err \
    bash -c "source ${VENV} && python -m src.utils.prepare_classification_cv --fold ${FOLD} --method standard"