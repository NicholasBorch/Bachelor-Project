# configs/classification_sce.py
# SCE-specific hyperparameters only.
# All shared parameters (epochs, lr, batch_size etc.)
# are defined in classification_default.py.

SCE_ALPHA: float = 0.1
SCE_BETA:  float = 1.0
SCE_A:     float = -4.0