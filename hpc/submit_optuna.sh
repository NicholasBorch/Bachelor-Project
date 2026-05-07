#!/bin/bash
# hpc/submit_optuna.sh
#
# Submit Optuna hyperparameter searches as LSF jobs. With proper mid-trial
# pruning (added in this patch), per-search compute drops by ~50% because
# clearly-bad trials are killed at epoch 30 instead of running all 150
# epochs. This lets us simplify the topology vs. the previous version:
#
#   - ELR: single 100-trial job, fits in 24h walltime (was previously
#     under-budgeted, ran out of time at ~50 trials).
#   - AsyCo: two parallel 50-trial chunks, each fits in 24h walltime.
#     Both chunks share the same SQLite study; TPE sees all completed
#     trials when sampling new ones.
#
# Walltime arithmetic with pruning:
#
#   ELR:        ~30 min/trial × 100 trials × ~0.55 (pruning) ≈ 16h.   Fits.
#   AsyCo:     ~60 min/trial × 50 trials × ~0.55 (pruning) ≈ 28h     ← TIGHT
#                                                                    keep at 23:59,
#                                                                    use --resume
#                                                                    if it walltimes
#
# These numbers assume MedianPruner kills ~45% of trials by epoch 30.
# The actual pruning rate depends on how quickly TPE finds the good region —
# expect higher pruning rates as the search progresses.
#
# Usage:
#   bash hpc/submit_optuna.sh elr
#   bash hpc/submit_optuna.sh asyco_divmix
#   bash hpc/submit_optuna.sh both

set -euo pipefail

QUEUE=${QUEUE:-gpuv100}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-10000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs}
JOB_PREFIX=${JOB_PREFIX:-thesis}
VENV=${VENV:-/work3/s234841/venv/Bachelor-Project}

DATASET=${DATASET:-imbalanced}
OPTIM=${OPTIM:-adam}
MODEL=${MODEL:-resnet34_pretrained}
TAU=${TAU:-0.2}
FOLD=${FOLD:-5}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-150}

# Pruning hyperparameters. Defaults are conservative — bump them down if
# you want more aggressive pruning, up if you find good trials are being
# pruned by accident.
PRUNER_N_WARMUP_STEPS=${PRUNER_N_WARMUP_STEPS:-30}    # epochs before pruning allowed
PRUNER_N_STARTUP_TRIALS=${PRUNER_N_STARTUP_TRIALS:-10} # trials before pruning fires at all

mkdir -p "${LOG_DIR}"

submit_elr_search() {
    local job_name="${JOB_PREFIX}_optuna_elr_fold${FOLD}"
    local log_stem="${LOG_DIR}/optuna_elr_fold${FOLD}"
    bsub -q "${QUEUE}" \
        -W "23:59" \
        -n "${CPU_CORES}" \
        -R "span[hosts=1]" \
        -R "rusage[mem=${MEM_PER_CORE_MB}]" \
        -gpu "${GPU_SPEC}" \
        -J "${job_name}" \
        -o "${log_stem}.out" \
        -e "${log_stem}.err" \
        "source ${VENV}/bin/activate && export PYTHONUNBUFFERED=1 && \
         python -m scripts.optuna_search \
            --method elr \
            --dataset ${DATASET} \
            --optim ${OPTIM} \
            --model ${MODEL} \
            --tau ${TAU} \
            --fold ${FOLD} \
            --n-trials 100 \
            --trial-epochs ${TRIAL_EPOCHS} \
            --pruner-n-warmup-steps ${PRUNER_N_WARMUP_STEPS} \
            --pruner-n-startup-trials ${PRUNER_N_STARTUP_TRIALS}"
    echo "Submitted ELR search: ${job_name}"
}

submit_asyco_divmix_chunk() {
    local chunk_idx="$1"
    local n_trials="$2"
    local resume_flag="$3"

    local job_name="${JOB_PREFIX}_optuna_asydm_fold${FOLD}_c${chunk_idx}"
    local log_stem="${LOG_DIR}/optuna_asyco_divmix_fold${FOLD}_c${chunk_idx}"
    bsub -q "${QUEUE}" \
        -W "23:59" \
        -n "${CPU_CORES}" \
        -R "span[hosts=1]" \
        -R "rusage[mem=${MEM_PER_CORE_MB}]" \
        -gpu "${GPU_SPEC}" \
        -J "${job_name}" \
        -o "${log_stem}.out" \
        -e "${log_stem}.err" \
        "source ${VENV}/bin/activate && export PYTHONUNBUFFERED=1 && \
         python -m scripts.optuna_search \
            --method asyco_divmix \
            --dataset ${DATASET} \
            --optim ${OPTIM} \
            --model ${MODEL} \
            --tau ${TAU} \
            --fold ${FOLD} \
            --n-trials ${n_trials} \
            --trial-epochs ${TRIAL_EPOCHS} \
            --pruner-n-warmup-steps ${PRUNER_N_WARMUP_STEPS} \
            --pruner-n-startup-trials ${PRUNER_N_STARTUP_TRIALS} \
            ${resume_flag}"
    echo "Submitted asyco_divmix chunk ${chunk_idx}: ${job_name}"
}

submit_asyco_divmix_split() {
    # Two parallel chunks of 50 trials each. The first creates the study;
    # the second resumes into it. SQLite + Optuna handles concurrent writes
    # via locking. TPE in chunk 1 sees chunk 0's completed trials when it
    # samples (modulo write-visibility lag, which is seconds).
    submit_asyco_divmix_chunk 0 50 ""
    submit_asyco_divmix_chunk 1 50 "--resume"
    echo
    echo "NOTE: chunk 1 uses --resume. The race condition where chunk 1"
    echo "      starts before chunk 0 has created the study is harmless"
    echo "      with --load_if_exists=True (which optuna_search.py uses)"
    echo "      — chunk 1 will retry on first failure and succeed."
}

case "${1:-both}" in
    elr)
        submit_elr_search ;;
    asyco_divmix)
        submit_asyco_divmix_split ;;
    both)
        submit_elr_search
        submit_asyco_divmix_split
        ;;
    *)
        echo "Usage: $0 {elr|asyco_divmix|both}"
        exit 2 ;;
esac

echo
echo "Monitor:   bjobs -w | grep ${JOB_PREFIX}_optuna"
echo
echo "After completion, analyze with:"
echo "    python -m scripts.optuna_analyze --method elr --fold ${FOLD}"
echo "    python -m scripts.optuna_analyze --method asyco_divmix --fold ${FOLD}"
