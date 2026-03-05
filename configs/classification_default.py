# Shared configuration for all classification experiments.

# Reproducibility
SEED = 10

# Cross-validation
OUTER_FOLDS = 5

# Noise injection
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]
NORM_STD = 0.10

# Image preprocessing
IMAGE_SIZE = 224

# DataLoader
BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True