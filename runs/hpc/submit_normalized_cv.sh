#!/bin/sh
# runs/hpc/submit_normalized_cv.sh

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

RUNNING=0
JOBIDS=()

for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "cvnorm${FOLD}" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 0:30 \
            -o logs/cvnorm_${FOLD}.out \
            -e logs/cvnorm_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method normalized \
            | awk '{print $2}' | tr -d '<>')
    else
        WAIT_ID=${JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "cvnorm${FOLD}" \
            -w "done(${WAIT_ID})" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 0:30 \
            -o logs/cvnorm_${FOLD}.out \
            -e logs/cvnorm_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method normalized \
            | awk '{print $2}' | tr -d '<>')
    fi
    JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD submitted: job $JOBID"
done

echo "CVNORM_JOBS=${JOBIDS[@]}"