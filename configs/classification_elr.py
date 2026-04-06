# configs/classification_elr.py
# ELR-specific hyperparameters only.
# All shared parameters (epochs, lr, batch_size etc.)
# are defined in classification_default.py.
#
# Based on Liu et al. (2020) "Early-Learning Regularization Prevents
# Memorization of Noisy Labels". NeurIPS 2020.
#
# Values follow the Clothing1M setup (pretrained ResNet-50, similar to ours):
#   beta   = 0.7  temporal ensembling momentum
#   lambda = 0.5  regularisation coefficient

ELR_BETA:   float = 0.7
ELR_LAMBDA: float = 0.5