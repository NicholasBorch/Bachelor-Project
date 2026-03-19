#!/bin/bash
# runs/hpc/noise_idn/master_noise_submit.sh

set -euo pipefail
cd $HOME/projects/Bachelor-Project
mkdir -p logs

echo "============================================"
echo "  Noise Preparation — 10-Fold CV"
echo "============================================"

# ── Step 1a: Standard IDN ─────────────────────────────────────────────────────
echo ""
echo "Step 1a: Standard IDN (10 folds)..."
STD_JOBIDS=()
for FOLD in $(seq 0 9); do
    JOBID=$(sed "s/--fold \$FOLD/--fold ${FOLD}/" runs/hpc/noise_idn/submit_standard_cv.sh \
        | bsub \
            -J "cvstd${FOLD}" \
            -oo logs/cvstd_${FOLD}.out \
            -eo logs/cvstd_${FOLD}.err \
        | awk '{print $2}' | tr -d '<>')
    STD_JOBIDS+=($JOBID)
    echo "  Fold $FOLD → job $JOBID"
done

# ── Step 1b: Normalised IDN ───────────────────────────────────────────────────
echo ""
echo "Step 1b: Normalised IDN (10 folds)..."
NORM_JOBIDS=()
for FOLD in $(seq 0 9); do
    JOBID=$(sed "s/--fold \$FOLD/--fold ${FOLD}/" runs/hpc/noise_idn/submit_normalized_cv.sh \
        | bsub \
            -J "cvnorm${FOLD}" \
            -oo logs/cvnorm_${FOLD}.out \
            -eo logs/cvnorm_${FOLD}.err \
        | awk '{print $2}' | tr -d '<>')
    NORM_JOBIDS+=($JOBID)
    echo "  Fold $FOLD → job $JOBID"
done

# ── Step 2: Fold prob collection ──────────────────────────────────────────────
echo ""
echo "Step 2: Fold prob collection (10 folds)..."
PROBS_JOBIDS=()
for FOLD in $(seq 0 9); do
    JOBID=$(sed "s/--fold \$FOLD/--fold ${FOLD}/" runs/hpc/noise_idn/submit_fold_probs.sh \
        | bsub \
            -J "foldprobs${FOLD}" \
            -oo logs/foldprobs_${FOLD}.out \
            -eo logs/foldprobs_${FOLD}.err \
        | awk '{print $2}' | tr -d '<>')
    PROBS_JOBIDS+=($JOBID)
    echo "  Fold $FOLD → job $JOBID"
done

# ── Step 3: Merge — waits for ALL 10 fold prob jobs ──────────────────────────
echo ""
echo "Step 3: Merge fold probs (waits for all fold prob jobs)..."

MERGE_DEPENDS=$(printf "done(%s)&&" "${PROBS_JOBIDS[@]}")
MERGE_DEPENDS=${MERGE_DEPENDS%&&}

MERGE_JOB=$(bsub \
    -J "mergeprobs" \
    -w "$MERGE_DEPENDS" \
    -oo logs/mergeprobs.out \
    -eo logs/mergeprobs.err \
    < runs/hpc/noise_idn/submit_merge_fold_probs.sh \
    | awk '{print $2}' | tr -d '<>')
echo "  Merge job → $MERGE_JOB"

# ── Step 4: Feature-driven IDN — waits for merge ─────────────────────────────
echo ""
echo "Step 4: Feature-driven IDN (10 folds, waits for merge)..."
FD_JOBIDS=()
for FOLD in $(seq 0 9); do
    JOBID=$(sed "s/--fold \$FOLD/--fold ${FOLD}/" runs/hpc/noise_idn/submit_feature_driven_cv.sh \
        | bsub \
            -J "cvfd${FOLD}" \
            -w "done(${MERGE_JOB})" \
            -oo logs/cvfd_${FOLD}.out \
            -eo logs/cvfd_${FOLD}.err \
        | awk '{print $2}' | tr -d '<>')
    FD_JOBIDS+=($JOBID)
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