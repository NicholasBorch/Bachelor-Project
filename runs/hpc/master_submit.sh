#!/bin/bash
# runs/hpc/master_submit.sh
#
# Submits the full noise preparation pipeline.
# Run once from repo root: bash runs/hpc/master_submit.sh
#
# Steps:
#   1a. Standard IDN       — 10 jobs, max 5 at once
#   1b. Normalised IDN     — 10 jobs, max 5 at once  (runs in parallel with 1a)
#   2.  Fold prob collect  — 10 jobs, max 5 at once  (runs in parallel with 1a/1b)
#   3.  Merge fold probs   — 1 job, waits for all of step 2
#   4.  Feature-driven IDN — 10 jobs, max 5 at once, waits for step 3

set -e
cd $HOME/projects/Bachelor-Project
mkdir -p logs

echo "============================================"
echo "  Noise Preparation — 10-Fold CV"
echo "============================================"

# ── Step 1a: Standard IDN ─────────────────────────────────────────────────────
echo ""
echo "Step 1a: Standard IDN (10 folds, max 5 parallel)..."
RUNNING=0
STD_JOBIDS=()
for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "cvstd${FOLD}" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" -W 0:30 \
            -o logs/cvstd_${FOLD}.out -e logs/cvstd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method standard \
            | awk '{print $2}' | tr -d '<>')
    else
        WAIT_ID=${STD_JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "cvstd${FOLD}" \
            -w "done(${WAIT_ID})" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" -W 0:30 \
            -o logs/cvstd_${FOLD}.out -e logs/cvstd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method standard \
            | awk '{print $2}' | tr -d '<>')
    fi
    STD_JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD → job $JOBID"
done

# ── Step 1b: Normalised IDN ───────────────────────────────────────────────────
echo ""
echo "Step 1b: Normalised IDN (10 folds, max 5 parallel)..."
RUNNING=0
NORM_JOBIDS=()
for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "cvnorm${FOLD}" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" -W 0:30 \
            -o logs/cvnorm_${FOLD}.out -e logs/cvnorm_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method normalized \
            | awk '{print $2}' | tr -d '<>')
    else
        WAIT_ID=${NORM_JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "cvnorm${FOLD}" \
            -w "done(${WAIT_ID})" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" -W 0:30 \
            -o logs/cvnorm_${FOLD}.out -e logs/cvnorm_${FOLD}.err \
            python -m src.utils.prepare_classification_cv --fold $FOLD --method normalized \
            | awk '{print $2}' | tr -d '<>')
    fi
    NORM_JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD → job $JOBID"
done

# ── Step 2: Fold prob collection ──────────────────────────────────────────────
echo ""
echo "Step 2: Fold prob collection (10 folds, max 5 parallel)..."
mkdir -p data/processed/HAM10000/fold_probs
RUNNING=0
PROBS_JOBIDS=()
for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "foldprobs${FOLD}" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=16GB]" \
            -gpu "num=1:mode=exclusive_process" -W 1:00 \
            -o logs/foldprobs_${FOLD}.out -e logs/foldprobs_${FOLD}.err \
            python -m src.utils.collect_fold_probs --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    else
        WAIT_ID=${PROBS_JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "foldprobs${FOLD}" \
            -w "done(${WAIT_ID})" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=16GB]" \
            -gpu "num=1:mode=exclusive_process" -W 1:00 \
            -o logs/foldprobs_${FOLD}.out -e logs/foldprobs_${FOLD}.err \
            python -m src.utils.collect_fold_probs --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    fi
    PROBS_JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD → job $JOBID"
done

# ── Step 3: Merge — waits for ALL 10 fold prob jobs ──────────────────────────
echo ""
echo "Step 3: Merge fold probs (waits for all fold prob jobs)..."

# Build dependency string: done(ID0)&&done(ID1)&&...&&done(ID9)
MERGE_DEPENDS=$(printf "done(%s)&&" "${PROBS_JOBIDS[@]}")
MERGE_DEPENDS=${MERGE_DEPENDS%&&}  # strip trailing &&

MERGE_JOB=$(bsub \
    -J "mergeprobs" \
    -w "$MERGE_DEPENDS" \
    -q hpc -n 1 -R "rusage[mem=8GB]" -W 0:10 \
    -o logs/mergeprobs.out -e logs/mergeprobs.err \
    python -m src.utils.merge_fold_probs \
    | awk '{print $2}' | tr -d '<>')
echo "  Merge job → $MERGE_JOB"

# ── Step 4: Feature-driven IDN — waits for merge ─────────────────────────────
echo ""
echo "Step 4: Feature-driven IDN (10 folds, max 5 parallel, waits for merge)..."
RUNNING=0
FD_JOBIDS=()
for FOLD in $(seq 0 9); do
    if [ $RUNNING -lt 5 ]; then
        JOBID=$(bsub \
            -J "cvfd${FOLD}" \
            -w "done(${MERGE_JOB})" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" -W 0:20 \
            -o logs/cvfd_${FOLD}.out -e logs/cvfd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv_feature_driven --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    else
        WAIT_ID=${FD_JOBIDS[$((RUNNING - 5))]}
        JOBID=$(bsub \
            -J "cvfd${FOLD}" \
            -w "done(${MERGE_JOB})&&done(${WAIT_ID})" \
            -q gpuv100 -n 4 -R "span[hosts=1]" -R "rusage[mem=8GB]" \
            -gpu "num=1:mode=exclusive_process" -W 0:20 \
            -o logs/cvfd_${FOLD}.out -e logs/cvfd_${FOLD}.err \
            python -m src.utils.prepare_classification_cv_feature_driven --fold $FOLD \
            | awk '{print $2}' | tr -d '<>')
    fi
    FD_JOBIDS+=($JOBID)
    RUNNING=$((RUNNING + 1))
    echo "  Fold $FOLD → job $JOBID"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  All jobs submitted. Monitor with: bjobs"
echo "============================================"
echo "  Standard IDN jobs    : ${STD_JOBIDS[@]}"
echo "  Normalised IDN jobs  : ${NORM_JOBIDS[@]}"
echo "  Fold prob jobs       : ${PROBS_JOBIDS[@]}"
echo "  Merge job            : ${MERGE_JOB}"
echo "  Feature-driven jobs  : ${FD_JOBIDS[@]}"
echo ""
echo "  Logs: logs/cvstd_*.out"
echo "        logs/cvnorm_*.out"
echo "        logs/foldprobs_*.out"
echo "        logs/mergeprobs.out"
echo "        logs/cvfd_*.out"
echo ""
echo "  Expected total wall time: ~80 minutes"