import numpy as np
import torch

from classifiers.PointTransformers.train_cls import fact
from registration.registration import reg_mpe


# TODO: Maybe, I should not have an EMD loss when doing binary classification.
# TODO: Why is the previous confusion matrices looking worse than the previous once I gathered?
#      Compare latex with the most previous confusion matrices. They should be run on the same model
#      but the results vastly differs.
# TODO: Cannot seem overlap_pred to work or Minkowski engine. Try some other method.
if __name__ == "__main__":
    if True:
        np.random.seed(1)
        # Set the seed for PyTorch
        torch.manual_seed(1)
        # If you are using CUDA (PyTorch with GPU support)
        torch.cuda.manual_seed(1)

    reg_mpe()