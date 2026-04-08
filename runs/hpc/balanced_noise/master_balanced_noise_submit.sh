#!/bin/bash
# master_balanced_noise_submit.sh
#
# Orchestrates all balanced-dataset noise preparation jobs with LSF dependency chaining.
# Mirror of runs/hpc/noise_idn/master_noise_submit.sh for the balanced experiment arm.
#
# Pipeline:
#   Step 1: Standard + Normalized IDN     (10 parallel, array job — single script, both variants)
#   Step 2: Balanced OOF prob collection  (10 parallel, array job — can run in parallel with Step 1)
#   Step 3: Merge OOF probs               (1 sequential, waits for ALL Step 2 jobs)
#   Step 4: Balanced feature-driven IDN   (10 parallel, waits for Step 3)
#
# Note: Standard IDN (cv_balanced_standard/) is created as a reference artefact only.
#       It is NOT used in training experiments. Normalized and feature-driven IDN are
#       the primary experiment noise types.
#
# Usage:
#   bash runs/hpc/balanced_noise/master_balanced_noise_submit.sh

set -euo pipefail
cd ~/projects/Bachelor-Project

SCRIPTS_DIR="runs/hpc/balanced_noise"
mkdir -p "${SCRIPTS_DIR}/logs"

echo "Submitting balanced noise preparation pipeline..."

# ── Step 1: Standard + Normalized IDN (10 parallel) ──────────────────────────
# A single job array writes both cv_balanced_standard/ and cv_balanced_normalized/
JID_IDN=$(bsub < "${SCRIPTS_DIR}/submit_balanced_normalized_cv.sh" | grep -oP '(?<=Job <)\d+')
echo "  Submitted standard + normalized IDN jobs — array JID=${JID_IDN}"

# ── Step 2: OOF prob collection (10 parallel, independent of Step 1) ──────────
JID_PROBS=$(bsub < "${SCRIPTS_DIR}/submit_balanced_fold_probs.sh" | grep -oP '(?<=Job <)\d+')
echo "  Submitted balanced OOF fold probs jobs — array JID=${JID_PROBS}"

# ── Step 3: Merge OOF probs (waits for ALL 10 step-2 jobs) ────────────────────
JID_MERGE=$(bsub -w "ended(${JID_PROBS})" \
    < "${SCRIPTS_DIR}/submit_balanced_merge_fold_probs.sh" \
    | grep -oP '(?<=Job <)\d+')
echo "  Submitted merge job (waits for JID=${JID_PROBS}) — JID=${JID_MERGE}"

# ── Step 4: Feature-driven IDN (waits for merge) ──────────────────────────────
JID_FD=$(bsub -w "done(${JID_MERGE})" \
    < "${SCRIPTS_DIR}/submit_balanced_feature_driven_cv.sh" \
    | grep -oP '(?<=Job <)\d+')
echo "  Submitted balanced feature-driven IDN jobs (waits for JID=${JID_MERGE}) — array JID=${JID_FD}"

echo ""
echo "Balanced noise preparation pipeline submitted."
echo "  Standard + Normalized IDN:  JID=${JID_IDN}    (10 array jobs)"
echo "  OOF probs:                  JID=${JID_PROBS}  (10 array jobs)"
echo "  Merge probs:                JID=${JID_MERGE}  (1 job, depends on ${JID_PROBS})"
echo "  Feature-driven IDN:         JID=${JID_FD}     (10 array jobs, depends on ${JID_MERGE})"
echo ""
echo "Monitor with: bjobs -u \$USER"
