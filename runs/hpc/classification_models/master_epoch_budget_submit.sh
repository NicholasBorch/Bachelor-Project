#!/bin/bash
# runs/hpc/classification_models/master_epoch_budget_submit.sh
#
# Submits 10 parallel fold jobs for epoch budget selection.
# After all complete, run aggregate_epoch_budget.py locally.
#
# Usage: bash runs/hpc/classification_models/master_epoch_budget_submit.sh

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs/classification_models

SCRIPT="runs/hpc/classification_models/submit_epoch_budget.sh"

echo "============================================"
echo "  Epoch Budget Selection — 10 Folds"
echo "============================================"

JOBIDS=()
for FOLD in $(seq 0 9); do
    JOBID=$(sed \
        -e "s/\$FOLD/${FOLD}/g" \
        "$SCRIPT" \
        | bsub \
            -J "epochbud${FOLD}" \
            -oo "logs/classification_models/epoch_budget_fold${FOLD}.out" \
            -eo "logs/classification_models/epoch_budget_fold${FOLD}.err" \
        | awk '{print $2}' | tr -d '<>')
    JOBIDS+=($JOBID)
    echo "  Fold ${FOLD} → job ${JOBID}"
done

echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Job IDs: ${JOBIDS[@]}"
echo ""
echo "  When all complete, run locally:"
echo "    python -m src.utils.aggregate_epoch_budget"