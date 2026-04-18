#!/bin/bash
# hpc/submit_stage3.sh
#
# Submit Stage 3 training jobs. Delegates to the Python generator so that
# filtering (e.g. "only AsyCo imbalanced") works identically for interactive
# resubmission and for bulk initial submission.
#
# Usage:
#   # All 1,920 jobs:
#   bash hpc/submit_stage3.sh
#
#   # Only one method:
#   bash hpc/submit_stage3.sh --method asyco
#
#   # Only one condition (useful for debugging / resubmission):
#   bash hpc/submit_stage3.sh --method elr --dataset imbalanced --init pretrained --optim sgd
#
#   # Dry-run: see the commands without submitting:
#   bash hpc/submit_stage3.sh --dry-run [other filters...]
#
# At 10-12 concurrent jobs with 24h walltime and ~1-2 hours per job median,
# the full campaign runs in 10-13 wall-clock days.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN=0
PYTHON_ARGS=()

# Strip out --dry-run; pass everything else to the generator
for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        PYTHON_ARGS+=("${arg}")
    fi
done

cd "${PROJECT_ROOT}"
mkdir -p logs

# Count first so the user sees what they are about to do
n=$(python -m hpc.generate_stage3_jobs "${PYTHON_ARGS[@]}" --count-only)
echo "Stage 3: ${n} jobs match your filters."

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry-run: printing bsub commands (not submitting)."
    python -m hpc.generate_stage3_jobs "${PYTHON_ARGS[@]}"
    exit 0
fi

read -rp "Submit ${n} jobs? [y/N] " reply
if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

python -m hpc.generate_stage3_jobs "${PYTHON_ARGS[@]}" | bash
echo "Submitted ${n} Stage 3 jobs."
echo "Monitor with: bjobs -w | grep thesis_"
