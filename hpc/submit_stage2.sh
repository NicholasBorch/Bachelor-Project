#!/bin/bash
# hpc/submit_stage2.sh
#
# Submit epoch-budget selection jobs. Grid: 2 datasets × 4 methods × 10
# folds = 80 jobs. Each trains the method on clean (τ=0) training folds
# for up to 100 epochs and logs validation metrics per epoch. Aggregation
# picks the median convergence epoch per (dataset, method).
#
# Walltime rationale (ResNet-34 ~1.5× faster than ResNet-50):
#
#                    obs @ 25 ep     @ 100 ep     walltime  margin
#   baseline/SCE/ELR balanced:    ~9 min         ~36 min       2:00     ~84 min
#   baseline/SCE/ELR imbalanced:  ~93 min        ~370 min      12:00    ~350 min
#   AsyCo balanced:                ~24 min        ~96 min       4:00     ~144 min
#   AsyCo imbalanced:              ~233 min       ~930 min      24:00    ~510 min
#
# (In practice Stage 2 usually converges well before 100 epochs, but the
# walltimes are sized for the worst case so that no selection run dies
# mid-convergence and wastes the fold.)
#
# Usage:
#   bash hpc/submit_stage2.sh

set -euo pipefail

QUEUE=${QUEUE:-gpuv100}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-10000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs}
JOB_PREFIX=${JOB_PREFIX:-thesis}

mkdir -p "${LOG_DIR}"

# Per (dataset, method) walltime resolution — generous margin on worst
# case to prevent silent truncation of the convergence tail.
get_walltime() {
    local dataset="$1"
    local method="$2"
    case "${dataset}-${method}" in
        balanced-baseline|balanced-sce|balanced-elr)    echo "2:30" ;;
        balanced-asyco)                                  echo "5:00" ;;
        imbalanced-baseline|imbalanced-sce|imbalanced-elr) echo "15:00" ;;
        imbalanced-asyco)                                 echo "23:59" ;;
        *) echo "23:59" ;;  # conservative fallback
    esac
}

submitted=0
for dataset in balanced imbalanced; do
    for method in baseline sce elr asyco; do
        walltime=$(get_walltime "${dataset}" "${method}")
        for fold in $(seq 0 9); do
            fold_padded=$(printf '%02d' "${fold}")
            job_name="${JOB_PREFIX}_stage2_${dataset}_${method}_fold${fold_padded}"
            log_stem="${LOG_DIR}/stage2_${dataset}_${method}_fold${fold_padded}"

            bsub -q "${QUEUE}" \
                -W "${walltime}" \
                -n "${CPU_CORES}" \
                -R "span[hosts=1]" \
                -R "rusage[mem=${MEM_PER_CORE_MB}]" \
                -gpu "${GPU_SPEC}" \
                -J "${job_name}" \
                -o "${log_stem}.out" \
                -e "${log_stem}.err" \
                "source /work3/s234841/venv/Bachelor-Project/bin/activate && export PYTHONUNBUFFERED=1 && python -m scripts.stage2_select_epoch_budget --dataset ${dataset} --method ${method} --fold ${fold}"
            submitted=$((submitted + 1))
        done
    done
done

echo "Submitted ${submitted} Stage 2 selection jobs (expected 80)."
echo
echo "Monitor:   bjobs -w | grep ${JOB_PREFIX}_stage2"
echo
echo "After all 80 jobs finish, aggregate per (dataset, method):"
echo "  for dataset in balanced imbalanced; do"
echo "    for method in baseline sce elr asyco; do"
echo "      python -m scripts.stage2_aggregate_epoch_budget --dataset \${dataset} --method \${method}"
echo "    done"
echo "  done"
