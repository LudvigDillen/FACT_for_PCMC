from classifiers.regression import perform_logistic_regression_training
from classifiers.regression import perform_logistic_regression_inference
from utils.data_handling import run_differential_entropy_on_dataset
from utils.parameters import Params


def coral(nusc, args, logger):
    params = Params(nusc=nusc, args=args, pointwise=True, do_fps=True)

    logger.info('Staring feature extraction CorAl')
    X_train, y_train, X_val, y_val, X_test, y_test = run_differential_entropy_on_dataset(params)

    # TRANING
    logger.info('Start training CorAl linear regression train...')
    model = perform_logistic_regression_training(X_train, y_train, X_val, y_val, logger,
                                                 epochs=200_000, learning_rate=0.03)

    accuracy_test = perform_logistic_regression_inference(X_test, y_test, model)

    return accuracy_test