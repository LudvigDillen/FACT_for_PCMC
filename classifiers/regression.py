"""
The code is based on Denny Loevlie's code (see below). (MIT license exists)
https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
https://gist.github.com/loevlie/5044e62aea2ce625b70d6d6d75113d25
"""


import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    # predict
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def perform_logistic_regression_training(
    X_train, y_train, X_val, y_val, logger, epochs=200_000, learning_rate=0.03
):
    """
    Perform logistic regression to predict alignment based on input features.

    Given input features x1 and x2, this function computes the logistic regression
    function to predict whether the data is aligned or misaligned, using the provided
    model parameters.

    Parameters:
    X (float): Consists of two features. Joint and separate diffential entropy. (joint first, sep second)
    y (bool): True or false regarding aligned or misaligned.

    param (tuple): Model parameters (beta0, beta1, beta2, threshold)
    """
    # Define hyperparameters
    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # Single binary output
    # Define logistic regression model
    model = LogisticRegression(input_dim, output_dim)
    # Define binary cross entropy loss
    criterion = torch.nn.BCELoss()
    # Define optimizer. Here Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Convart data to tensors

    X_train, X_val = torch.Tensor(X_train), torch.Tensor(X_val)
    y_train, y_val = torch.Tensor(y_train), torch.Tensor(y_val)

    losses = []
    losses_val = []
    accuracy = []
    best_accuracy_val = 0
    best_epoch = 0
    for epoch in tqdm(range(int(epochs)), desc="Training Epochs"):
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), y_train)

        loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias

        optimizer.step()  # Updates weights and biases with the optimizer (SGD)
        with torch.inference_mode():
            # Calculating the loss and accuracy for the test dataset
            # Get z for test data
            outputs_val = torch.squeeze(model(X_val))
            # Get loss for test data
            loss_val = criterion(outputs_val, y_val)
            # Get predictions for test data. If I want I could change the threshold here ...
            predicted_val = outputs_val.round().detach().numpy()
            total_val = y_val.size(0)
            # Get number of correct tests
            correct_val = np.sum(predicted_val == y_val.detach().numpy())
            # Get accuracy test
            accuracy_val = correct_val / total_val
            losses_val.append(loss_val.item())

            # Calculating the loss and accuracy for the train dataset
            # Get predictions for train data. If I want I could change the threshold here ...
            predicted_train = torch.squeeze(outputs).round().detach().numpy()
            total = y_train.size(0)
            # Get number of correct train examples
            correct = np.sum(predicted_train == y_train.detach().numpy())
            # Get accuracy train
            accuracy = correct / total
            losses.append(loss.item())

            if accuracy_val >= best_accuracy_val:
                best_epoch = epoch + 1
                best_accuracy_val = accuracy_val

                savepath = "best_model.pth"
                state = {
                    "epoch": best_epoch,
                    "instance_acc": best_accuracy_val,
                    "train_acc": accuracy,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
            last_epoch = (epoch + 1) == int(epochs)
            if not (epoch + 1) % 2500 or last_epoch:
                logger.info(f"Iteration: {epoch+1}")
                logger.info(
                    f"Loss [Train|Val]: [{loss.item():.4f}|{loss_val.item():.4f}]"
                )
                logger.info(f"Acc. [Train|Val]: [{accuracy:.2f}|{accuracy_val:.2f}]")
    # Load the best validation model
    best_model_path = "best_model.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def perform_logistic_regression_inference(X, y, model):
    """
    Runs inference using logistic regression model to determine if data features
    indicate true or false.

    Parameters:
    X (float): Consists of two features. Joint and separate diffential entropy.
    (joint first, sep second)

    accuracy_test (float)
    """
    # Convert data to tensors
    X, y = torch.Tensor(X), torch.Tensor(y)

    # Get z for test data
    outputs_test = torch.squeeze(model(X))
    # Get predictions for test data. If I want I could change the threshold here ...
    predicted_test = outputs_test.round().detach().numpy()
    total_test = y.size(0)
    # Get number of correct tests
    correct_test = np.sum(predicted_test == y.detach().numpy())
    # Get accuracy test
    accuracy_test = correct_test / total_test

    return accuracy_test, predicted_test


# minimal definition of a custom scipy distribution.
class CLogLog(stats.rv_continuous):
    def _ppf(self, q):
        return np.log(-np.log(1 - q))

    def _cdf(self, x):
        return 1 - np.exp(-np.exp(x))


def multinomial_ordinal_regression(X_train, y_train, X_val, y_val, logger):
    # Convert data to pandas DataFrame
    labels = np.hstack((y_train, y_val))[:, None]
    features = np.vstack((X_train, X_val))
    data = pd.DataFrame(
        np.hstack((labels, features)),
        columns=["label", "Joint Differential Entropy", "Separate Differential Entropy"])

    # Ensure the label column is treated as an ordered categorical variable
    data['label'] = pd.Categorical(data['label'], ordered=True)
    # Define and fit the ordinal regression model
    res_prob = OrderedModel(data['label'],
                            data[["Joint Differential Entropy", "Separate Differential Entropy"]],
                            distr="logit").fit(method='bfgs')
    logger.info(res_prob.summary())
    return res_prob
