import numpy as np
import torch

from classifiers.PointTransformers.train_cls import fact
from registration.registration import reg_mpe
from registration.geotransformer_handling import geotransformer_with_fact
from misc.metrics import metrics_vs_gt_class


if __name__ == "__main__":
    if True:
        np.random.seed(1)
        # Set the seed for PyTorch
        torch.manual_seed(1)
        # If you are using CUDA (PyTorch with GPU support)
        torch.cuda.manual_seed(1)

    fact()
    #metrics_vs_gt_class()
    # reg_mpe()
