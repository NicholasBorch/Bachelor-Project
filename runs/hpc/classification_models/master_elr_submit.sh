#!/bin/bash
# runs/hpc/classification_models/master_elr_submit.sh
#
# Submits ELR classification jobs for normalized and feature-driven noise types.
# Run once from repo root:
#   bash runs/hpc/classification_models/master_elr_submit.sh
#
# Structure: one job per fold per noise type = 20 jobs total.
# Completed runs are skipped automatically on resubmission.

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs/classification_models

NOISE_TYPES=("normalized_idn" "feature_driven_idn")
SCRIPT="runs/hpc/classification_models/submit_elr.sh"

echo "============================================"
echo "  ELR Classification — Normalized & Feature-Driven"
echo "============================================"

for NOISE_TYPE in "${NOISE_TYPES[@]}"; do
    echo ""
    echo "Submitting noise_type=${NOISE_TYPE} (10 folds)..."
    for FOLD in $(seq 0 9); do
        JOBID=$(sed \
            -e "s/\$FOLD/${FOLD}/g" \
            -e "s/\$NOISE_TYPE/${NOISE_TYPE}/g" \
            "$SCRIPT" \
            | bsub \
                -J "elr${NOISE_TYPE:0:3}${FOLD}" \
                -oo "logs/classification_models/elr_${NOISE_TYPE}_fold${FOLD}.out" \
                -eo "logs/classification_models/elr_${NOISE_TYPE}_fold${FOLD}.err" \
            | awk '{print $2}' | tr -d '<>')
        echo "  fold=${FOLD} → job ${JOBID}"
    done
done

echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Logs: logs/classification_models/elr_*.out"
echo ""
echo "  To check completion:"
echo "    find results/HAM10000/elr -name 'test_metrics.json' | wc -l"
echo "  Expected: 140 (2 noise types x 10 folds x 7 tau levels)"