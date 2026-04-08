# configs/classification_balanced.py
#
# Configuration flag for balanced dataset experiments.
# All training hyperparameters are inherited from classification_default.py.
# This file exists to make the "balanced" intent explicit in imports/logs.
#
# Usage in runner:
#   from configs.classification_balanced import BALANCED
#
# The BALANCED flag is used by run_balanced_classification_cv.py to pass
# use_weighted_sampler=False to all method runners.

BALANCED = True  # Flag: use shuffle=True, no WeightedRandomSampler, no class-weighted CE

# The metadata path override is handled in the data-prep scripts, not here.
# The CV directory override is handled in run_balanced_classification_cv.py
# via NOISE_TYPE_TO_CV_DIR.
