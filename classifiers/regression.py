import torch


def logistic_regression(x1, x2, param):
    """
    Perform logistic regression to predict alignment based on input features.

    Given input features x1 and x2, this function computes the logistic regression
    function to predict whether the data is aligned or misaligned, using the provided
    model parameters.

    Parameters:
    x1 (float): First input feature (differential entropy of joint point cloud)
    x2 (float): Second input feature (differential entropy of separate point clouds)
    param (tuple): Model parameters (beta0, beta1, beta2, threshold)

    Returns:
    int: Predicted label (0 for misaligned, 1 for aligned)
    """
    # Unpack the model parameters
    beta0, beta1, beta2, th = param

    # Compute the linear combination of input features and model parameters
    z = beta0 + beta1 * x1 + beta2 * x2

    # Calculate the probability using the logistic function
    p = 1 / (1 + torch.exp(-z))

    # Make a prediction based on the probability and threshold
    if p < th:
        y_pred = 0  # misaligned
    else:
        y_pred = 1  # aligned

    return y_pred
