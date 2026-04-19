#!/bin/bash
# hpc/submit_stage1b.sh
#
# Submit OOF probability collection jobs, one per (dataset, fold).
# 2 datasets × 10 folds = 20 jobs total.
#
# Each job trains ResNet-18 for 30 epochs with Adam(lr=1e-4) on the
# clean training folds and runs inference on the held-out fold.
# Locked protocol — see PROJECT_DOCUMENTATION.md §6 Stage 1b / §9.
#
# Prerequisite: Stage 1a (fold assignments) has been run locally.
#
# Usage:
#   bash hpc/submit_stage1b.sh
#
# LSF site settings align with hpc/lsf_defaults.yaml and the DTU HPC guide.
# Edit lsf_defaults.yaml first if you move clusters.
#
# Walltime rationale (30 epochs, ResNet-18, ~2× faster than ResNet-34):
#   - Balanced (~460 train samples):   ~4-6 min per job
#   - Imbalanced (~6,700 train samples): ~50-60 min per job
#   With 3:00 walltime -> ~120 min safety margin on imbalanced jobs.

set -euo pipefail

QUEUE=${QUEUE:-gpuv100}
WALLTIME=${WALLTIME:-3:00}
CPU_CORES=${CPU_CORES:-8}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-4000}
GPU_SPEC=${GPU_SPEC:-"num=1:mode=exclusive_process"}
LOG_DIR=${LOG_DIR:-logs}
JOB_PREFIX=${JOB_PREFIX:-thesis}

mkdir -p "${LOG_DIR}"

submitted=0
for dataset in balanced imbalanced; do
    for fold in $(seq 0 9); do
        fold_padded=$(printf '%02d' "${fold}")
        job_name="${JOB_PREFIX}_stage1b_${dataset}_fold${fold_padded}"
        log_stem="${LOG_DIR}/stage1b_${dataset}_fold${fold_padded}"

        bsub -q "${QUEUE}" \
            -W "${WALLTIME}" \
            -n "${CPU_CORES}" \
            -R "span[hosts=1]" \
            -R "rusage[mem=${MEM_PER_CORE_MB}]" \
            -gpu "${GPU_SPEC}" \
            -J "${job_name}" \
            -o "${log_stem}.out" \
            -e "${log_stem}.err" \
            "source /work3/s234841/venv/Bachelor-Project/bin/activate && export PYTHONUNBUFFERED=1 && python -m scripts.stage1b_collect_oof_probs --dataset ${dataset} --fold ${fold}"
        submitted=$((submitted + 1))
    done
done

echo "Submitted ${submitted} Stage 1b jobs (expected 20)."
echo
echo "Monitor:   bjobs -w | grep ${JOB_PREFIX}_stage1b"
echo "After all finish, run locally on the HPC login node:"
echo "    python -m scripts.stage1b_merge_oof_probs --dataset balanced"
echo "    python -m scripts.stage1b_merge_oof_probs --dataset imbalanced"
