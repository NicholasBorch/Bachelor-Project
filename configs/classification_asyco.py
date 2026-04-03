# configs/classification_asyco.py
#
# Hyperparameters for AsyCo classification experiments.
#
# WARMUP_EPOCHS: Number of epochs of full supervised training (standard CE
# for clf_net, BCE for ref_net) before the sample selection and relabeling
# mechanism activates. The original paper uses ~10-20% of total training.
# With our fixed 25-epoch budget, 3 warmup epochs (~12%) leaves 22 epochs
# for the AsyCo mechanism to operate, which is proportionally aligned with
# the paper's regime. The previous value of 10 consumed 40% of training,
# leaving too few post-warmup epochs for selection to stabilise before
# cosine annealing decayed the learning rate toward zero.

WARMUP_EPOCHS = 3     # epochs of full supervised training before selection begins
K_TOPLABEL    = 2     # number of top-ranked labels for reference net prediction
LAMBDA_U      = 25.0  # weight of unsupervised consistency loss on noisy samples
TEMPERATURE   = 0.5   # sharpening temperature for pseudo-labels