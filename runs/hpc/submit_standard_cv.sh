#!/bin/sh
# runs/hpc/submit_standard_cv.sh
# Submits 10 individual jobs for standard IDN, max 5 running at once.
# Submit from repo root: bash runs/hpc/submit_standard_cv.sh

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

RUNNING=0
JOBIDS=()

for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "cvstd${FOLD}" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 0:30 \
            -o logs/cvstd_${FOLD}.out \
            -e logs/cvstd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method standard \
            | awk '{print $2}' | tr -d '<>')
    else
        # Wait for the job 5 slots ago to keep max 5 running
        WAIT_ID=${JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "cvstd${FOLD}" \
            -w "done(${WAIT_ID})" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 0:30 \
            -o logs/cvstd_${FOLD}.out \
            -e logs/cvstd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method standard \
            | awk '{print $2}' | tr -d '<>')
    fi
    JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD submitted: job $JOBID"
done

# Return all job IDs as a done condition string for downstream dependencies
echo "CVSTD_JOBS=${JOBIDS[@]}"