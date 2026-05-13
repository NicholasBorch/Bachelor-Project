#!/bin/bash
# hpc/submit_final_experiment.sh
#
# Submit the Main Experiment to LSF. The 240-job grid is:
#   methods (4) x dataset (1) x init (1) x optim (1) x tau (6) x fold (10)
#
# All knobs are env-var-overridable so future re-runs (e.g. scratch or sgd)
# require no code changes:
#
#   DATASET=imbalanced               # the dataset to run on
#   INIT=pretrained                  # resnet34_{init} initialization
#   OPTIM=adam                       # optimizer
#   METHODS="baseline sce elr asyco_divmix"
#   TAUS="0.0 0.1 0.2 0.3 0.4 0.5"
#   FOLDS="0 1 2 3 4 5 6 7 8 9"
#   TUNING_FOLD=5                    # which fold the Optuna search ran on
#   TUNING_TAU=0.2                   # which tau the Optuna search trained on
#
# Examples:
#   bash hpc/submit_final_experiment.sh
#
#   # Re-run scratch later:
#   INIT=scratch bash hpc/submit_final_experiment.sh
#
#   # Only AsyCo, only the first 3 folds (smoke test):
#   METHODS=asyco_divmix FOLDS="0 1 2" bash hpc/submit_final_experiment.sh
#
# SUBMISSION ORDER (matches user spec):
#   For each method in order [baseline sce elr asyco_divmix]:
#     For each fold in 0..9:
#       For each tau in 0.0 0.1 ... 0.5:
#         submit 1 job
# So queue order is: all 60 baseline jobs (fold 0 tau 0.0..0.5, fold 1 ..., ...),
# then all 60 SCE jobs, then all 60 ELR jobs, then all 60 AsyCo jobs.
#
# WALLTIMES (generous, ~5x expected runtime):
#   baseline       2:30   (expected ~30 min/job at 150 epochs)
#   sce            2:30   (expected ~30 min/job)
#   elr            3:00   (expected ~35 min/job)
#   asyco_divmix   5:00   (expected ~60 min/job)
#
# Idempotent: jobs whose test_metrics.json already exists are skipped by the
# Python entry point.

set -euo pipefail

QUEUE=${QUEUE:-gpuv100}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-10000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs/final_experiment}
JOB_PREFIX=${JOB_PREFIX:-thesis_FINAL}
VENV=${VENV:-/work3/s234841/venv/Bachelor-Project}

DATASET=${DATASET:-imbalanced}
INIT=${INIT:-pretrained}
OPTIM=${OPTIM:-adam}
METHODS=${METHODS:-"baseline sce elr asyco_divmix"}
TAUS=${TAUS:-"0.0 0.1 0.2 0.3 0.4 0.5"}
FOLDS=${FOLDS:-"0 1 2 3 4 5 6 7 8 9"}
TUNING_FOLD=${TUNING_FOLD:-5}
TUNING_TAU=${TUNING_TAU:-0.2}

mkdir -p "${LOG_DIR}"


walltime_for_method() {
    case "$1" in
        baseline)     echo "2:30" ;;
        sce)          echo "2:30" ;;
        elr)          echo "3:00" ;;
        asyco_divmix) echo "5:00" ;;
        *)
            echo "ERROR: unknown method '$1', cannot determine walltime" >&2
            exit 1
            ;;
    esac
}


# tau=0.1 -> "tau10" for the job name
tau_tag() {
    printf "tau%02d" "$(awk -v t="$1" 'BEGIN{printf "%.0f", t*100}')"
}


submit_one_job() {
    # Args: method, fold, tau
    local method="$1"
    local fold="$2"
    local tau="$3"
    local walltime
    walltime=$(walltime_for_method "${method}")

    local job_name="${JOB_PREFIX}_${method}_$(tau_tag "${tau}")_fold${fold}"
    local log_stem="${LOG_DIR}/${method}_$(tau_tag "${tau}")_fold${fold}"

    bsub -q "${QUEUE}" \
        -W "${walltime}" \
        -n "${CPU_CORES}" \
        -R "span[hosts=1]" \
        -R "rusage[mem=${MEM_PER_CORE_MB}]" \
        -gpu "${GPU_SPEC}" \
        -J "${job_name}" \
        -o "${log_stem}.out" \
        -e "${log_stem}.err" \
        "source ${VENV}/bin/activate && export PYTHONUNBUFFERED=1 && \
         python -m scripts.final_experiment_train \
            --method ${method} \
            --dataset ${DATASET} \
            --init ${INIT} \
            --optim ${OPTIM} \
            --tau ${tau} \
            --fold ${fold} \
            --tuning-fold ${TUNING_FOLD} \
            --tuning-tau ${TUNING_TAU}" \
        > /dev/null
}


echo "=== Submitting Main Experiment ==="
echo "Dataset:       ${DATASET}"
echo "Init:          ${INIT}"
echo "Optim:         ${OPTIM}"
echo "Methods:       ${METHODS}"
echo "Taus:          ${TAUS}"
echo "Folds:         ${FOLDS}"
echo "Tuning fold:   ${TUNING_FOLD} (from FINAL Optuna search)"
echo "Tuning tau:    ${TUNING_TAU}"
echo

total_submitted=0
for method in ${METHODS}; do
    method_count=0
    walltime=$(walltime_for_method "${method}")
    echo "Submitting ${method} (walltime ${walltime} per job, fold-major order):"
    for fold in ${FOLDS}; do
        for tau in ${TAUS}; do
            submit_one_job "${method}" "${fold}" "${tau}"
            method_count=$((method_count + 1))
            total_submitted=$((total_submitted + 1))
        done
    done
    echo "  ${method}: ${method_count} jobs submitted"
done

echo
echo "Total jobs submitted: ${total_submitted}"
echo
echo "Monitor:"
echo "  bjobs -w | grep ${JOB_PREFIX}"
echo "  bjobs -w | grep ${JOB_PREFIX} | awk '{print \$3}' | sort | uniq -c   # state breakdown"
echo
echo "Status check (after some jobs complete):"
echo "  python -m scripts.final_experiment_status"
echo
echo "Analyze (after all jobs complete):"
echo "  python -m scripts.final_experiment_analyze"
