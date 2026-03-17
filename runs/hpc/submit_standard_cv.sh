#!/bin/sh
# runs/hpc/submit_standard_cv.sh
#BSUB -J cv_standard[0-9]%5
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 10GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o logs/cv_standard_%J_%I.out
#BSUB -e logs/cv_standard_%J_%I.err

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

echo "=== Standard IDN CV | Fold $LSB_JOBINDEX ==="
python -m src.utils.prepare_classification_cv --fold $LSB_JOBINDEX --method standard
echo "=== Done ==="