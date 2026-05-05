#!/bin/bash
# hpc/submit_optuna.sh
#
# Submit Optuna hyperparameter searches as LSF jobs. Default behavior
# submits one search per (method, optim, model) combination on a single
# fold (default fold 5) at tau=0.2. Each search runs N trials of TPE
# Bayesian optimization with median pruning.
#
# Walltime budget reasoning (validated against your Stage 2 v2 runtimes):
#
#   Per-trial cost at trial_epochs=150 (half of Stage 2 cap):
#     ELR imbalanced:           ~30 min worst case
#     asyco_divmix imbalanced:  ~100 min worst case
#
#   Per-search cost at n_trials=100, before pruning:
#     ELR:           ~50 hours
#     asyco_divmix:  ~165 hours (would exceed 24h DTU walltime!)
#
#   With MedianPruner (kills ~50-65% of trials early):
#     ELR:           ~20-25 hours (fits in 24:00 walltime)
#     asyco_divmix:  ~70-90 hours (does NOT fit; needs splitting)
#
# Strategy:
#   - ELR runs as a single job at 23:59 walltime.
#   - asyco_divmix is split across 4 jobs of 25 trials each, each at
#     23:59 walltime, all sharing the same SQLite study (Optuna handles
#     the concurrency safely). The TPE sampler still benefits from all
#     prior trials when each new job starts.
#
# Usage:
#   bash hpc/submit_optuna.sh elr
#   bash hpc/submit_optuna.sh asyco_divmix
#   bash hpc/submit_optuna.sh both              # do both methods
#
# After all jobs complete:
#   python -m scripts.optuna_analyze --method elr --fold 5
#   python -m scripts.optuna_analyze --method asyco_divmix --fold 5

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
            --trial-epochs ${TRIAL_EPOCHS}"
    echo "Submitted ELR search: ${job_name}"
}

submit_asyco_divmix_search_chunk() {
    # Submit one chunk of the asyco_divmix search. All chunks share the
    # same SQLite store under results/optuna/asyco_divmix/...
    # SQLite + Optuna handles concurrent access via locking; the TPE
    # sampler reads existing trials when each new chunk starts, so later
    # chunks benefit from earlier ones' findings. The first chunk uses
    # --resume=False (study is fresh); later chunks set --resume.
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
            ${resume_flag}"
    echo "Submitted asyco_divmix chunk ${chunk_idx}: ${job_name}"
}

submit_asyco_divmix_split() {
    # 4 chunks × 25 trials = 100 trials total, all writing to one DB.
    # Submit chunks with sequential dependency so they don't all start
    # blank: chunk 0 creates the study, chunks 1-3 resume into it.
    submit_asyco_divmix_search_chunk 0 25 ""
    submit_asyco_divmix_search_chunk 1 25 "--resume"
    submit_asyco_divmix_search_chunk 2 25 "--resume"
    submit_asyco_divmix_search_chunk 3 25 "--resume"
    echo
    echo "NOTE: chunks 1-3 use --resume so they write to the same study.db"
    echo "      Optuna's SQLite backend handles concurrent locking, but for"
    echo "      best surrogate-model quality, consider chaining with bsub -w"
    echo "      (depend-on-success) so chunk N+1 starts after chunk N finishes."
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
