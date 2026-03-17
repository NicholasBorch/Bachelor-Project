#!/bin/sh
# runs/hpc/submit_normalized_cv.sh
#BSUB -J cv_normalized[0-9]%5
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 10GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o logs/cv_normalized_%J_%I.out
#BSUB -e logs/cv_normalized_%J_%I.err

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

echo "=== Normalised IDN CV | Fold $LSB_JOBINDEX ==="
python -m src.utils.prepare_classification_cv --fold $LSB_JOBINDEX --method normalized
echo "=== Done ==="