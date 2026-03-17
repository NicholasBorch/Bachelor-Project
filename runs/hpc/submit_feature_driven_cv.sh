#!/bin/sh
# runs/hpc/submit_feature_driven_cv.sh
# Called by master_submit.sh after fold_probs_merge completes.
# Manual: bsub -w "done(fold_probs_merge)" < runs/hpc/submit_feature_driven_cv.sh
#BSUB -J cv_feature_driven[0-9]%5
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 10GB
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:20
#BSUB -o logs/cv_feature_driven_%J_%I.out
#BSUB -e logs/cv_feature_driven_%J_%I.err

cd $HOME/projects/Bachelor-Project
source .venv/bin/activate
mkdir -p logs

echo "=== Feature-Driven IDN CV | Fold $LSB_JOBINDEX ==="
python -m src.utils.prepare_classification_cv_feature_driven --fold $LSB_JOBINDEX
echo "=== Done ==="