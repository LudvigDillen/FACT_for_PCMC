import os
import numpy as np

from classifiers.regression import perform_logistic_regression_training
from classifiers.regression import perform_logistic_regression_inference
from utils.data_handling import run_differential_entropy_on_dataset
from utils.parameters import Params


def get_coral_features(nusc, args, logger):
    params = Params(nusc=nusc, args=args, pointwise=True)

    logger.info("Staring feature extraction CorAl...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = run_differential_entropy_on_dataset(params, logger)
    logger.info("Feature extraction finished")
    if not os.path.exists(args.feature_folder):
        os.makedirs(args.feature_folder)
    np.savetxt(args.feature_folder + "/X_train.txt", X_train)
    np.savetxt(args.feature_folder + "/y_train.txt", y_train)
    np.savetxt(args.feature_folder + "/X_val.txt", X_val)
    np.savetxt(args.feature_folder + "/y_val.txt", y_val)
    np.savetxt(args.feature_folder + "/X_test.txt", X_test)
    np.savetxt(args.feature_folder + "/y_test.txt", y_test)
    logger.info(f"Features saved to {args.feature_folder}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def perform_coral_training(
    X_train, y_train, X_val, y_val, logger, epochs=10_000, learning_rate=0.03
):
    # TRANING
    logger.info("Start training CorAl linear regression train...")
    model = perform_logistic_regression_training(
        X_train,
        y_train,
        X_val,
        y_val,
        logger,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    logger.info("Finsih training CorAl linear regression train")
    return model


def perform_coral_inference(X_test, y_test, model):
    accuracy_test, predicted_test = perform_logistic_regression_inference(
        X_test, y_test, model
    )
    return accuracy_test, predicted_test
