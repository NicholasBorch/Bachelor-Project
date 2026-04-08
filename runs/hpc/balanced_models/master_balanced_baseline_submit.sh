#!/bin/bash
# runs/hpc/balanced_models/master_balanced_baseline_submit.sh
#
# Submits balanced baseline classification jobs for normalized and feature-driven noise types.
# Run once from repo root:
#   bash runs/hpc/balanced_models/master_balanced_baseline_submit.sh
#
# Structure: one job per fold per noise type = 20 jobs total.
# Set EPOCHS below before submitting (25 or 50).

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs/balanced_models

EPOCHS=25   # ← set to 25 or 50 before submitting
NOISE_TYPES=("balanced_normalized_idn" "balanced_feature_driven_idn")
SCRIPT="runs/hpc/balanced_models/submit_balanced_baseline.sh"

echo "============================================"
echo "  Balanced Baseline — Normalized & Feature-Driven"
echo "  Epochs: ${EPOCHS}"
echo "============================================"

for NOISE_TYPE in "${NOISE_TYPES[@]}"; do
    echo ""
    echo "Submitting noise_type=${NOISE_TYPE} (10 folds)..."
    for FOLD in $(seq 0 9); do
        JOBID=$(sed \
            -e "s/\$FOLD/${FOLD}/g" \
            -e "s/\$NOISE_TYPE/${NOISE_TYPE}/g" \
            -e "s/\$EPOCHS/${EPOCHS}/g" \
            "$SCRIPT" \
            | bsub \
                -J "bal_base_${NOISE_TYPE:9:3}_ep${EPOCHS}_f${FOLD}" \
                -oo "logs/balanced_models/baseline_${NOISE_TYPE}_ep${EPOCHS}_fold${FOLD}.out" \
                -eo "logs/balanced_models/baseline_${NOISE_TYPE}_ep${EPOCHS}_fold${FOLD}.err" \
            | awk '{print $2}' | tr -d '<>')
        echo "  fold=${FOLD} → job ${JOBID}"
    done
done

echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Logs: logs/balanced_models/baseline_*.out"
echo ""
echo "  To check completion:"
echo "    find results/HAM10000/baseline/balanced_normalized_idn -name 'test_metrics.json' | wc -l"
echo "    find results/HAM10000/baseline/balanced_feature_driven_idn -name 'test_metrics.json' | wc -l"
echo "  Expected: 70 each (10 folds x 7 tau levels)"
