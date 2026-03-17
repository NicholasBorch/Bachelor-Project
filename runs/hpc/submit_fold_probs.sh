#!/bin/sh
# runs/hpc/submit_fold_probs.sh
#BSUB -J fold_probs[0-9]%5
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 18GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -o logs/fold_probs_%J_%I.out
#BSUB -e logs/fold_probs_%J_%I.err

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs
mkdir -p data/processed/HAM10000/fold_probs

echo "=== Fold Prob Collection | Fold $LSB_JOBINDEX ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
python -m src.utils.collect_fold_probs --fold $LSB_JOBINDEX
echo "=== Done ==="