#!/bin/sh
# runs/hpc/submit_merge_fold_probs.sh
# Not submitted directly — called by master_submit.sh with dependency set.
# Manual submission: bsub -w "done(fold_probs)" < runs/hpc/submit_merge_fold_probs.sh
#BSUB -J fold_probs_merge
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -W 0:10
#BSUB -o logs/fold_probs_merge_%J.out
#BSUB -e logs/fold_probs_merge_%J.err

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

echo "=== Merging Fold Probability Files ==="
python -m src.utils.merge_fold_probs
echo "=== Merge complete ==="