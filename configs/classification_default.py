# THIS IS EXAMPLE DRAFT

# Cross-validation
N_FOLDS = 10
SEED = 42

# Dataset
DATASET = "HAM10000"

# Noise
NOISE_TYPE = "idn"
NOISE_RATE = 0.2

# Method
METHOD = "baseline"

# Model
MODEL_NAME = "resnet18"
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Dropout / MC dropout
USE_MC_DROPOUT = False
MC_SAMPLES = 10