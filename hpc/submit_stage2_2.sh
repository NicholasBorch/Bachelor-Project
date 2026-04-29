#!/bin/bash
# hpc/submit_stage2_2.sh
#
# Submit Stage 2 v2 epoch-budget selection jobs across optimizer / model-init
# configurations. Grid:
#
#   2 datasets × 5 methods × 2 optimizers × 2 model configs × 10 folds
#   = 400 jobs
#
# Each job trains the method on clean (τ=0) training folds for up to the
# epoch cap specified in config (now 300 in base.yaml), logging validation
# metrics per epoch for later aggregation / analysis.
#
# This follows the original Stage 2 setup closely, but expands it to run:
#   - SGD + pretrained ResNet-34
#   - Adam + pretrained ResNet-34
#   - SGD + scratch ResNet-34
#   - Adam + scratch ResNet-34
#
# Usage:
#   bash hpc/submit_stage2_2.sh

set -euo pipefail

QUEUE=${QUEUE:-gpuv100}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-8000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs}
JOB_PREFIX=${JOB_PREFIX:-thesis}

mkdir -p "${LOG_DIR}"

# Conservative walltimes for 300-epoch runs.
# Scratch/pretrained should be similar in runtime; Adam may be slightly slower.
# AsyCo+DivMix entries get ~2× the comparable AsyCo budget because MixMatch
# adds two more clf_net forward passes per step plus a 2× larger MixUp batch.
get_walltime() {
    local dataset="$1"
    local method="$2"
    local optim="$3"
    local model="$4"

    case "${dataset}-${method}-${optim}-${model}" in
        balanced-asyco_divmix-*-*)         echo "1:30" ;;
        balanced-*-*-*)                    echo "0:40" ;;
        imbalanced-asyco_divmix-sgd-*)     echo "5:00" ;;
        imbalanced-asyco_divmix-adam-*)    echo "6:00" ;;
        imbalanced-*-sgd-*)                echo "1:30" ;;
        imbalanced-*-adam-*)               echo "2:30" ;;
        *)                                 echo "2:30" ;;
    esac
}

submitted=0
for dataset in balanced imbalanced; do
    for method in baseline sce elr asyco asyco_divmix; do
        for optim in sgd adam; do
            for model in resnet34_pretrained resnet34_scratch; do
                walltime=$(get_walltime "${dataset}" "${method}" "${optim}" "${model}")
                for fold in $(seq 0 9); do
                    fold_padded=$(printf '%02d' "${fold}")
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
                        "source /work3/s234841/venv/Bachelor-Project/bin/activate && export PYTHONUNBUFFERED=1 && python -m scripts.stage2_select_epoch_budget_2 --dataset ${dataset} --method ${method} --optim ${optim} --model ${model} --fold ${fold}"
                    submitted=$((submitted + 1))
                done
            done
        done
    done
done

echo "Submitted ${submitted} Stage 2 v2 selection jobs (expected 400)."
echo
echo "Monitor:   bjobs -w | grep ${JOB_PREFIX}_stage2_2"
echo
echo "Outputs are written under:"
echo "  results/epoch_selection_v2/{dataset}/{method}/{optim}/{model}/fold_XX/"
echo
echo "Manifests are written under:"
echo "  results/manifests/stage2_select_2_*.json"
