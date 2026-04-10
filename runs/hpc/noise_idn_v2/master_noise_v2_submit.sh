#!/bin/bash
# runs/hpc/noise_idn_v2/master_noise_v2_submit.sh
#
# Submits feature-driven IDN v2 (argmax variant) fold CSV generation
# for the full imbalanced dataset.
#
# Requires fold_probs/fold_probs_full.npy to already exist from the
# original noise preparation pipeline (master_noise_submit.sh).
# No new OOF collection is needed — v2 reuses the same OOF probs as v1.
#
# Usage:
#   bash runs/hpc/noise_idn_v2/master_noise_v2_submit.sh

set -euo pipefail
cd ~/projects/Bachelor-Project

SCRIPTS_DIR="runs/hpc/noise_idn_v2"
mkdir -p "${SCRIPTS_DIR}/logs"

# Check that OOF probs exist
PROBS_FILE="data/processed/HAM10000/fold_probs/fold_probs_full.npy"
if [ ! -f "${PROBS_FILE}" ]; then
    echo "ERROR: OOF probs not found at ${PROBS_FILE}"
    echo "Run the original noise preparation pipeline first:"
    echo "  bash runs/hpc/noise_idn/master_noise_submit.sh"
    echo "  (wait for completion, then re-run this script)"
    exit 1
fi

echo "============================================"
echo "  Feature-Driven IDN v2 (Argmax Variant)"
echo "  Dataset: full imbalanced (7,470 samples)"
echo "  OOF probs: ${PROBS_FILE}"
echo "============================================"

JID=$(bsub < "${SCRIPTS_DIR}/submit_feature_driven_v2.sh" \
    | grep -oP '(?<=Job <)\d+')
echo "  Submitted v2 fold generation — array JID=${JID}"

echo ""
echo "Monitor with: bjobs -u \$USER"
echo "Logs: ${SCRIPTS_DIR}/logs/fd_v2_fold_*.out"
echo ""
echo "After completion, run analysis locally:"
echo "  python -m src.utils.analyze_idn_v2"
