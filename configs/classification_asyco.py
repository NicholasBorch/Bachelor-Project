# Hyperparameters for AsyCo classification experiments.

WARMUP_EPOCHS = 10    # epochs of full supervised training before selection begins
K_TOPLABEL    = 2     # number of top-ranked labels for reference net prediction
LAMBDA_U      = 25.0  # weight of unsupervised consistency loss on noisy samples
TEMPERATURE   = 0.5   # sharpening temperature for pseudo-labels