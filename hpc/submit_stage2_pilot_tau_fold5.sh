#!/bin/bash
# hpc/submit_stage2_pilot_tau_fold5.sh
#
# Pilot:
#   2 datasets × 2 methods × 2 optims × 2 models × 2 tau levels × 1 fold
#   = 32 jobs
#
# Runs fold 5 for tau=0.0 and tau=0.2.
# Training uses noisy labels for tau=0.2.
# Validation uses clean labels.
#
# Outputs:
#   results/pilot_stage2_tau_fold5/tau_00/...
#   results/pilot_stage2_tau_fold5/tau_20/...

set -euo pipefail

QUEUE=${QUEUE:-gpuv100}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-8000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs}
JOB_PREFIX=${JOB_PREFIX:-thesis}
FOLD=${FOLD:-5}

mkdir -p "${LOG_DIR}"

get_walltime() {
    local dataset="$1"
    local method="$2"
    local optim="$3"
    local model="$4"
    local tau="$5"

    case "${dataset}-${method}-${optim}-${model}-${tau}" in
        balanced-asyco_divmix-*-*-*)      echo "1:00" ;;
        balanced-*-*-*-*)                 echo "0:30" ;;
        imbalanced-asyco_divmix-*-*-*)    echo "4:30" ;;
        imbalanced-*-*-*-*)               echo "2:00" ;;
        *)                                echo "2:30" ;;
    esac
}

submitted=0

for dataset in balanced imbalanced; do
    for method in elr asyco_divmix; do
        for optim in sgd adam; do
            for model in resnet34_pretrained resnet34_scratch; do
                for tau in 0.0 0.2; do
                    tau_tag=$(python - <<PY
tau = float("${tau}")
print(f"tau_{int(round(tau * 100)):02d}")
PY
)
                    fold_padded=$(printf '%02d' "${FOLD}")
                    walltime=$(get_walltime "${dataset}" "${method}" "${optim}" "${model}" "${tau}")

                    job_name="${JOB_PREFIX}_pilot_s2_${tau_tag}_${dataset}_${method}_${optim}_${model}_fold${fold_padded}"
                    log_stem="${LOG_DIR}/pilot_s2_${tau_tag}_${dataset}_${method}_${optim}_${model}_fold${fold_padded}"

                    bsub -q "${QUEUE}" \
                        -W "${walltime}" \
                        -n "${CPU_CORES}" \
                        -R "span[hosts=1]" \
                        -R "rusage[mem=${MEM_PER_CORE_MB}]" \
                        -gpu "${GPU_SPEC}" \
                        -J "${job_name}" \
                        -o "${log_stem}.out" \
                        -e "${log_stem}.err" \
                        "source /work3/s234841/venv/Bachelor-Project/bin/activate && export PYTHONUNBUFFERED=1 && python -m scripts.stage2_select_epoch_budget_pilot_tau --dataset ${dataset} --method ${method} --optim ${optim} --model ${model} --tau ${tau} --fold ${FOLD}"

                    submitted=$((submitted + 1))
                done
            done
        done
    done
done

echo "Submitted ${submitted} Stage 2 tau pilot jobs (expected 32)."
echo "Monitor: bjobs -w | grep ${JOB_PREFIX}_pilot_s2"
echo "Outputs: results/pilot_stage2_tau_fold5/{tau_00,tau_20}/..."