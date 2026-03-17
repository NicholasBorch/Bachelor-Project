#!/bin/sh
# runs/hpc/submit_merge_fold_probs.sh
# Called from master_submit.sh with dependency string passed in.
# Manual: bash runs/hpc/submit_merge_fold_probs.sh "done(ID1)&&done(ID2)&&..."

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

DEPENDS=$1  # dependency string passed from master_submit.sh

if [ -z "$DEPENDS" ]; then
    JOBID=$(bsub \
        -J "mergeprobs" \
        -q hpc \
        -n 1 \
        -R "rusage[mem=8GB]" \
        -W 0:10 \
        -o logs/mergeprobs.out \
        -e logs/mergeprobs.err \
        python -m src.utils.merge_fold_probs \
        | awk '{print $2}' | tr -d '<>')
else
    JOBID=$(bsub \
        -J "mergeprobs" \
        -w "$DEPENDS" \
        -q hpc \
        -n 1 \
        -R "rusage[mem=8GB]" \
        -W 0:10 \
        -o logs/mergeprobs.out \
        -e logs/mergeprobs.err \
        python -m src.utils.merge_fold_probs \
        | awk '{print $2}' | tr -d '<>')
fi

echo "  Merge job submitted: $JOBID"
echo "MERGE_JOB=$JOBID"