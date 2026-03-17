#!/bin/bash
# runs/hpc/master_submit.sh
#
# Chains all noise preparation steps in order using LSF job dependencies.
# Run once from the repo root:
#   bash runs/hpc/master_submit.sh
#
# Execution order:
#   Step 1a (parallel, 10 jobs, max 5 at once): Standard IDN
#   Step 1b (parallel, 10 jobs, max 5 at once): Normalised IDN
#   Step 2  (parallel, 10 jobs, max 5 at once): Fold prob collection
#   Step 3  (single job, waits for step 2):     Merge fold probs
#   Step 4  (parallel, 10 jobs, max 5 at once): Feature-driven IDN

set -e
cd $HOME/projects/Bachelor-Project
mkdir -p logs

echo "============================================"
echo "  Noise Preparation — 10-Fold CV"
echo "============================================"

echo ""
echo "Step 1a: Standard IDN (10 folds, max 5 parallel)..."
STD_JOB=$(bsub < runs/hpc/submit_standard_cv.sh | awk '{print $2}' | tr -d '<>')
echo "  Job ID: $STD_JOB"

echo ""
echo "Step 1b: Normalised IDN (10 folds, max 5 parallel)..."
NORM_JOB=$(bsub < runs/hpc/submit_normalized_cv.sh | awk '{print $2}' | tr -d '<>')
echo "  Job ID: $NORM_JOB"

echo ""
echo "Step 2: Fold prob collection (10 folds, max 5 parallel)..."
PROBS_JOB=$(bsub < runs/hpc/submit_fold_probs.sh | awk '{print $2}' | tr -d '<>')
echo "  Job ID: $PROBS_JOB"

echo ""
echo "Step 3: Merge fold probs (waits for fold_probs to finish)..."
MERGE_JOB=$(bsub -w "done(fold_probs)" < runs/hpc/submit_merge_fold_probs.sh \
    | awk '{print $2}' | tr -d '<>')
echo "  Job ID: $MERGE_JOB"

echo ""
echo "Step 4: Feature-driven IDN (10 folds, max 5 parallel, waits for merge)..."
FD_JOB=$(bsub -w "done(fold_probs_merge)" < runs/hpc/submit_feature_driven_cv.sh \
    | awk '{print $2}' | tr -d '<>')
echo "  Job ID: $FD_JOB"

echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Standard IDN     : $STD_JOB"
echo "  Normalised IDN   : $NORM_JOB"
echo "  Fold probs       : $PROBS_JOB"
echo "  Fold probs merge : $MERGE_JOB"
echo "  Feature-driven   : $FD_JOB"
echo ""
echo "  Logs: logs/cv_standard_*.out"
echo "        logs/cv_normalized_*.out"
echo "        logs/fold_probs_*.out"
echo "        logs/fold_probs_merge_*.out"
echo "        logs/cv_feature_driven_*.out"
echo ""
echo "  Expected total wall time: ~80 minutes"