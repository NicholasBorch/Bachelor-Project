#!/bin/sh
# runs/hpc/submit_fold_probs.sh

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs
mkdir -p data/processed/HAM10000/fold_probs

RUNNING=0
JOBIDS=()

for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "foldprobs${FOLD}" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=16GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 1:00 \
            -o logs/foldprobs_${FOLD}.out \
            -e logs/foldprobs_${FOLD}.err \
            python -m src.utils.collect_fold_probs --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    else
        WAIT_ID=${JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "foldprobs${FOLD}" \
            -w "done(${WAIT_ID})" \
            -q gpuv100 \
            -n 4 \
            -R "span[hosts=1]" \
            -R "rusage[mem=16GB]" \
            -gpu "num=1:mode=exclusive_process" \
            -W 1:00 \
            -o logs/foldprobs_${FOLD}.out \
            -e logs/foldprobs_${FOLD}.err \
            python -m src.utils.collect_fold_probs --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    fi
    JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD submitted: job $JOBID"
done

echo "FOLDPROBS_JOBS=${JOBIDS[@]}"