#!/bin/sh
# runs/hpc/submit_feature_driven_cv.sh
# Called from master_submit.sh with merge job ID passed in.
# Manual: bash runs/hpc/submit_feature_driven_cv.sh <merge_job_id>

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

MERGE_ID=$1  # merge job ID passed from master_submit.sh
RUNNING=0
JOBIDS=()

for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        # First 5 folds wait only for merge to complete
        JOBID=$(bsub \
            -J "cvfd${FOLD}" \
            -w "done(${MERGE_ID})" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 0:20 \
            -o logs/cvfd_${FOLD}.out \
            -e logs/cvfd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv_feature_driven --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    else
        # Folds 5-9 wait for merge AND for the job 5 slots ago
        WAIT_ID=${JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "cvfd${FOLD}" \
            -w "done(${MERGE_ID})&&done(${WAIT_ID})" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 0:20 \
            -o logs/cvfd_${FOLD}.out \
            -e logs/cvfd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv_feature_driven --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    fi
    JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD submitted: job $JOBID"
done