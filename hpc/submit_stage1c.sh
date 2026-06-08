#!/bin/bash
# hpc/submit_stage1c.sh
#
# Submit noise injection jobs: 2 datasets x 3 noise types x 10 folds = 60.
# Each job processes all 6 tau values for one (dataset, noise_type, fold).
# Fast and CPU-bound (queue `hpc`); can also be run locally on the login node.

set -euo pipefail

QUEUE=${QUEUE:-hpc}
WALLTIME=${WALLTIME:-2:00}         # ~5 min real; 1h gives massive margin
CPU_CORES=${CPU_CORES:-4}
MEM_PER_CORE_MB=${MEM_PER_CORE_MB:-16000}
LOG_DIR=${LOG_DIR:-logs}
JOB_PREFIX=${JOB_PREFIX:-thesis}

mkdir -p "${LOG_DIR}"

submitted=0
for dataset in balanced imbalanced; do
    for noise_type in standard normalized feature_driven; do
        for fold in $(seq 0 9); do
            fold_padded=$(printf '%02d' "${fold}")
            job_name="${JOB_PREFIX}_stage1c_${dataset}_${noise_type}_fold${fold_padded}"
            log_stem="${LOG_DIR}/stage1c_${dataset}_${noise_type}_fold${fold_padded}"

            bsub -q "${QUEUE}" \
                -W "${WALLTIME}" \
                -n "${CPU_CORES}" \
                -R "span[hosts=1]" \
                -R "rusage[mem=${MEM_PER_CORE_MB}]" \
                -J "${job_name}" \
                -o "${log_stem}.out" \
                -e "${log_stem}.err" \
                "source /work3/s234841/venv/Bachelor-Project/bin/activate && export PYTHONUNBUFFERED=1 && python -m scripts.stage1c_inject_noise --dataset ${dataset} --noise-type ${noise_type} --fold ${fold}"
            submitted=$((submitted + 1))
        done
    done
done

echo "Submitted ${submitted} Stage 1c jobs (expected 60)."
echo
echo "Monitor:   bjobs -w | grep ${JOB_PREFIX}_stage1c"
echo
echo "After all finish, you may want to run on the HPC login node:"
echo "    python -m scripts.stage1d_characterize_noise"
echo "    python -m scripts.stage1e_human_comparison"