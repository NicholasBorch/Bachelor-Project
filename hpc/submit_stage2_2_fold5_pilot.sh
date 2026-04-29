#!/bin/bash
# hpc/submit_stage2_2_fold5_pilot.sh
#
# Pilot Stage 2 v2 run:
# 2 datasets × 2 methods × 2 optimizers × 2 model configs × 1 fold = 16 jobs

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

    case "${dataset}-${method}-${optim}-${model}" in
        balanced-asyco_divmix-*-*)         echo "1:00" ;;
        balanced-*-*-*)                    echo "0:30" ;;
        imbalanced-asyco_divmix-*-*)       echo "6:30" ;;
        imbalanced-*-*)                    echo "2:00" ;;
        *)                                 echo "2:30" ;;
    esac
}

submitted=0

for dataset in balanced imbalanced; do
    for method in elr asyco_divmix; do
        for optim in sgd adam; do
            for model in resnet34_pretrained resnet34_scratch; do
                walltime=$(get_walltime "${dataset}" "${method}" "${optim}" "${model}")
                fold_padded=$(printf '%02d' "${FOLD}")

                job_name="${JOB_PREFIX}_stage2_2_${dataset}_${method}_${optim}_${model}_fold${fold_padded}"
                log_stem="${LOG_DIR}/stage2_2_${dataset}_${method}_${optim}_${model}_fold${fold_padded}"

                bsub -q "${QUEUE}" \
                    -W "${walltime}" \
                    -n "${CPU_CORES}" \
                    -R "span[hosts=1]" \
                    -R "rusage[mem=${MEM_PER_CORE_MB}]" \
                    -gpu "${GPU_SPEC}" \
                    -J "${job_name}" \
                    -o "${log_stem}.out" \
                    -e "${log_stem}.err" \
                    "source /work3/s234841/venv/Bachelor-Project/bin/activate && export PYTHONUNBUFFERED=1 && python -m scripts.stage2_select_epoch_budget_2 --dataset ${dataset} --method ${method} --optim ${optim} --model ${model} --fold ${FOLD}"

                submitted=$((submitted + 1))
            done
        done
    done
done

echo "Submitted ${submitted} Stage 2 v2 pilot jobs (expected 16)."
echo "Monitor: bjobs -w | grep ${JOB_PREFIX}_stage2_2"