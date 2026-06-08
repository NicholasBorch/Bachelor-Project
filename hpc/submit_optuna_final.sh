#!/bin/bash
# Submit the final Optuna hyperparameter search for ELR, SCE, and AsyCo+DivMix.
# 7 jobs (ELR 2x50, SCE 2x50, AsyCo 34+33+33), 150 epochs/trial, tuning fold 5,
# train tau=0.2, clean validation labels. Three method chains run in parallel;
# chunks within a chain are sequential via bsub -w ended(JOBID).
# Outputs: results/optuna_final/{method}/imbalanced/adam_resnet34_pretrained/tau_20/fold_05/
#
# After all jobs finish, analyze with:
#   python -m scripts.stage2_tune_analyze --method elr          --fold 5
#   python -m scripts.stage2_tune_analyze --method sce          --fold 5
#   python -m scripts.stage2_tune_analyze --method asyco_divmix --fold 5

set -euo pipefail
 
QUEUE=${QUEUE:-gpuv100}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-10000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs/optuna_final}
JOB_PREFIX=${JOB_PREFIX:-thesis_optunaFINAL}
VENV=${VENV:-/work3/s234841/venv/Bachelor-Project}
 
DATASET=${DATASET:-imbalanced}
OPTIM=${OPTIM:-adam}
MODEL=${MODEL:-resnet34_pretrained}
TAU=${TAU:-0.2}
FOLD=${FOLD:-5}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-150}
PRUNER_N_WARMUP_STEPS=${PRUNER_N_WARMUP_STEPS:-30}
PRUNER_N_STARTUP_TRIALS=${PRUNER_N_STARTUP_TRIALS:-10}
 
mkdir -p "${LOG_DIR}"
 
 
# Submit one chunk (method, chunk_idx, n_trials, dep_jobid); prints job id to stdout.
submit_chunk() {
    local method="$1"
    local chunk_idx="$2"
    local n_trials="$3"
    local dep_jobid="$4"
 
    local resume_flag=""
    if [ "${chunk_idx}" -gt 1 ]; then
        resume_flag="--resume"
    fi
 
    local dep_arg=""
    if [ -n "${dep_jobid}" ]; then
        dep_arg="-w ended(${dep_jobid})"
    fi
 
    local job_name="${JOB_PREFIX}_${method}_fold${FOLD}_j${chunk_idx}"
    local log_stem="${LOG_DIR}/optuna_final_${method}_fold${FOLD}_j${chunk_idx}"
 
    local bsub_output
    bsub_output=$(bsub -q "${QUEUE}" \
        -W "23:59" \
        -n "${CPU_CORES}" \
        -R "span[hosts=1]" \
        -R "rusage[mem=${MEM_PER_CORE_MB}]" \
        -gpu "${GPU_SPEC}" \
        ${dep_arg} \
        -J "${job_name}" \
        -o "${log_stem}.out" \
        -e "${log_stem}.err" \
        "source ${VENV}/bin/activate && export PYTHONUNBUFFERED=1 && \
         python -m scripts.stage2_tune_search \
            --method ${method} \
            --dataset ${DATASET} \
            --optim ${OPTIM} \
            --model ${MODEL} \
            --tau ${TAU} \
            --fold ${FOLD} \
            --n-trials ${n_trials} \
            --trial-epochs ${TRIAL_EPOCHS} \
            --pruner-n-warmup-steps ${PRUNER_N_WARMUP_STEPS} \
            --pruner-n-startup-trials ${PRUNER_N_STARTUP_TRIALS} \
            ${resume_flag}")
 
    # bsub output format: "Job <12345> is submitted to queue <gpuv100>."
    local job_id
    job_id=$(echo "${bsub_output}" | grep -oP 'Job <\K[0-9]+')
 
    if [ -z "${job_id}" ]; then
        echo "ERROR: failed to extract job id from bsub output:" >&2
        echo "${bsub_output}" >&2
        exit 1
    fi
 
    if [ -n "${dep_jobid}" ]; then
        echo "Submitted ${job_name} (job id ${job_id}) — depends on ${dep_jobid}" >&2
    else
        echo "Submitted ${job_name} (job id ${job_id})" >&2
    fi
    echo "${job_id}"
}
 
 
echo "=== FINAL Optuna search submission ===" >&2
echo "Tuning fold: ${FOLD}" >&2
echo "Submission order: AsyCo1, SCE1, ELR1, AsyCo2, SCE2, ELR2, AsyCo3" >&2
echo >&2
 
# Order matters — the bsub queue position will follow submission order.
asyco_j1=$(submit_chunk asyco_divmix 1 34 "")
sce_j1=$(submit_chunk sce          1 50 "")
elr_j1=$(submit_chunk elr          1 50 "")
asyco_j2=$(submit_chunk asyco_divmix 2 33 "${asyco_j1}")
sce_j2=$(submit_chunk sce          2 50 "${sce_j1}")
elr_j2=$(submit_chunk elr          2 50 "${elr_j1}")
asyco_j3=$(submit_chunk asyco_divmix 3 33 "${asyco_j2}")
 
echo >&2
echo "All 7 jobs submitted." >&2
echo >&2
echo "Job ids:" >&2
echo "  asyco_divmix: ${asyco_j1} -> ${asyco_j2} -> ${asyco_j3}" >&2
echo "  sce:          ${sce_j1} -> ${sce_j2}" >&2
echo "  elr:          ${elr_j1} -> ${elr_j2}" >&2
echo >&2
echo "Monitor:   bjobs -w | grep ${JOB_PREFIX}" >&2
echo >&2
echo "After completion, analyze with:" >&2
echo "    python -m scripts.stage2_tune_analyze --method elr          --fold ${FOLD}" >&2
echo "    python -m scripts.stage2_tune_analyze --method sce          --fold ${FOLD}" >&2
echo "    python -m scripts.stage2_tune_analyze --method asyco_divmix --fold ${FOLD}" >&2