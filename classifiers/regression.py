import torch
from tqdm import tqdm
import numpy as np

"""
Kudos to Denny Loevlie from whom I've got code inspiration from. See link. (MIT license exists)
https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
https://gist.github.com/loevlie/5044e62aea2ce625b70d6d6d75113d25
"""


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    # predict
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def perform_logistic_regression(
        X_train, X_test, y_train, y_test, epochs=200_000, learning_rate=0.03, verbose=True):
    """
    Perform logistic regression to predict alignment based on input features.

    Given input features x1 and x2, this function computes the logistic regression
    function to predict whether the data is aligned or misaligned, using the provided
    model parameters.

    Parameters:
    x1 (float): First input feature (differential entropy of joint point cloud)
    x2 (float): Second input feature (differential entropy of separate point clouds)
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
    # TODO: Maybe better to use ADAM e.g.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Convart data to tensors
    # TODO: Should I move the computations to GPU (cuda)?
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    losses = []
    losses_test = []
    accuracy_test = 0
    # for epoch in tqdm(range(1, int(epochs)+1), desc='Training Epochs'):
    for epoch in range(1, int(epochs)+1):
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), y_train)

        loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias

        optimizer.step()  # Updates weights and biases with the optimizer (SGD)
        last_epoch = (epoch == int(epochs))
        if epoch % 5000 == 0 or last_epoch:
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                # Get z for test data
                outputs_test = torch.squeeze(model(X_test))
                # Get loss for test data
                loss_test = criterion(outputs_test, y_test)
                # Get predictions for test data. If I want I could change the threshold here ...
                predicted_test = outputs_test.round().detach().numpy()
                total_test = y_test.size(0)
                # Get number of correct tests
                correct_test = np.sum(predicted_test == y_test.detach().numpy())
                # Get accuracy test
                accuracy_test = 100 * correct_test/total_test
                losses_test.append(loss_test.item())

                # Calculating the loss and accuracy for the train dataset
                # Get predictions for train data. If I want I could change the threshold here ...
                predicted_train = torch.squeeze(outputs).round().detach().numpy()
                total = y_train.size(0)
                # Get number of correct train examples
                correct = np.sum(predicted_train == y_train.detach().numpy())
                # Get accuracy train
                accuracy = 100 * correct/total
                losses.append(loss.item())
                if verbose:
                    loss_train_print = np.around(loss.item(), 4)
                    loss_test_print = np.around(loss_test.item(), 4)
                    accuracy_train_print = np.around(accuracy, 2)
                    accuracy_test_print = np.around(accuracy_test, 2)
                    print(f"Iteration: {epoch}", flush=True)
                    print((f"[Train|Test]: Loss = [{loss_train_print}|{loss_test_print}]",
                           f"Acc. = [{accuracy_train_print}|{accuracy_test_print}]"), flush=True)
    return model, accuracy_test
