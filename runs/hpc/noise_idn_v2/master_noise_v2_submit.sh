#!/bin/bash
# runs/hpc/noise_idn_v2/master_noise_v2_submit.sh
#
# Feature-driven IDN v2 (argmax variant) — standalone submission.
#
# Reuses data/processed/HAM10000/fold_probs/fold_probs_full.npy from the
# original master_noise_submit.sh pipeline. No prob collection or merge
# is re-run. Only the 10 feature-driven v2 fold prep jobs are submitted.

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs

echo "============================================"
echo "  Noise Preparation v2 — Feature-Driven (argmax)"
echo "============================================"

PROBS_FILE="data/processed/HAM10000/fold_probs/fold_probs_full.npy"
if [ ! -f "$PROBS_FILE" ]; then
    echo "ERROR: $PROBS_FILE not found."
    echo "Run runs/hpc/noise_idn/master_noise_submit.sh first to produce fold probs."
    exit 1
fi
echo "Found existing fold probs: $PROBS_FILE"

# ── Feature-driven IDN v2 (10 folds) ──────────────────────────────────────────
echo ""
echo "Submitting feature-driven v2 (10 folds)..."
FD2_JOBIDS=()
for FOLD in $(seq 0 9); do
    JOBID=$(sed "s/--fold \$FOLD/--fold ${FOLD}/" runs/hpc/noise_idn_v2/submit_feature_driven_v2_cv.sh \
        | bsub \
            -J "cvfd2${FOLD}" \
            -oo logs/cvfd2_${FOLD}.out \
            -eo logs/cvfd2_${FOLD}.err \
        | awk '{print $2}' | tr -d '<>')
    FD2_JOBIDS+=($JOBID)
    echo "  Fold $FOLD → job $JOBID"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Feature-driven v2 jobs : ${FD2_JOBIDS[@]}"
echo ""
echo "  Logs: logs/cvfd2_*.out"
echo ""
echo "  Output: data/processed/HAM10000/cv_feature_driven_v2/"
