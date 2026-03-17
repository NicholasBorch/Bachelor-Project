#!/bin/bash
# Manual resubmission of the merge job.
# Usage: bash runs/hpc/submit_merge_fold_probs.sh

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs
VENV="$HOME/projects/Bachelor-Project/.venv/bin/activate"

bsub \
    -J "mergeprobs" \
    -q hpc \
    -n 1 \
    -R "rusage[mem=8000]" \
    -W 0:10 \
    -oo logs/mergeprobs.out \
    -eo logs/mergeprobs.err \
    bash -c "source ${VENV} && python -m src.utils.merge_fold_probs"