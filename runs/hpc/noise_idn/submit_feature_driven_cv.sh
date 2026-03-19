#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 02:00

set -euo pipefail
cd ~/projects/Bachelor-Project
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m src.utils.prepare_classification_cv_feature_driven --fold $FOLD