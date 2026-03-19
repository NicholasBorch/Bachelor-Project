#!/bin/bash
# runs/hpc/classification_models/master_classification_submit.sh
#
# Submits baseline classification jobs for all folds and noise types.
# Run once from repo root:
#   bash runs/hpc/classification_models/master_classification_submit.sh
#
# Structure: one job per fold per noise type = 30 jobs total.
# Each job processes all 7 tau levels sequentially within the fold.
# Wall time is set to 23:59 to accommodate 100 epochs x 7 tau levels.
# Completed runs are skipped automatically on resubmission.

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs/classification_models

NOISE_TYPES=("standard_idn" "normalized_idn" "feature_driven_idn")
SCRIPT="runs/hpc/classification_models/submit_baseline.sh"

echo "============================================"
echo "  Baseline Classification — All Noise Types"
echo "============================================"

ALL_JOBIDS=()

for NOISE_TYPE in "${NOISE_TYPES[@]}"; do
    echo ""
    echo "Submitting noise_type=${NOISE_TYPE} (10 folds)..."
    for FOLD in $(seq 0 9); do
        JOBID=$(sed \
            -e "s/\$FOLD/${FOLD}/g" \
            -e "s/\$NOISE_TYPE/${NOISE_TYPE}/g" \
            "$SCRIPT" \
            | bsub \
                -J "base${FOLD}${NOISE_TYPE:0:3}" \
                -oo "logs/classification_models/baseline_${NOISE_TYPE}_fold${FOLD}.out" \
                -eo "logs/classification_models/baseline_${NOISE_TYPE}_fold${FOLD}.err" \
            | awk '{print $2}' | tr -d '<>')
        ALL_JOBIDS+=($JOBID)
        echo "  fold=${FOLD} → job ${JOBID}"
    done
done

echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Total jobs submitted: ${#ALL_JOBIDS[@]}"
echo "  All job IDs: ${ALL_JOBIDS[@]}"
echo ""
echo "  Logs: logs/classification_models/"
echo ""
echo "  To check for failures after completion:"
echo "    grep -l 'Error\|Traceback' logs/classification_models/*.err"
echo ""
echo "  Expected wall time per job: up to 24 hours"