# Shared configuration for all classification experiments.

# Reproducibility
SEED = 10

# Cross-validation
OUTER_FOLDS = 5

# Noise injection
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NORM_STD = 0.10

# Image preprocessing
IMAGE_SIZE = 224

# DataLoader
BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True

# Training
EPOCHS         = 100
LR             = 1e-4
BACKBONE_DEPTH = 50