import numpy as np
import torch

from classifiers.PointTransformers.train_cls import main as fact
from registration.registration import reg, compare_reg_methods


# TODO: Maybe, I should not have an EMD loss when doing binary classification.
# TODO: WIP to create other distribution for data I do classification on, either (see utils.pointclouds.py)
#           - GMM
#           - registration
# TODO: Check so that the registration code still works. I changed a bit but haven't checked that it works still.
# TODO: Commit soon ...
if __name__ == "__main__":
    if True:
        np.random.seed(1)
        # Set the seed for PyTorch
        torch.manual_seed(1)
        # If you are using CUDA (PyTorch with GPU support)
        torch.cuda.manual_seed(1)

    fact()