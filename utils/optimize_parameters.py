from ax import optimize

from features.differential_entropy import differential_entropy_test_accuracy
from utils.data_handling import gather_data
from utils.nuscenes_handling import read_nuscenes_data


def optimize_with_ax(samples_training=5, samples_test=5, verbose=False, total_trials=20):
    PC_scenes = read_nuscenes_data()
    PC_scenes_training, PC_scenes_test = gather_data(
        PC_scenes, samples_training=samples_training, samples_test=samples_test)

    def evaluation_function_wrapper(params):
        objective = differential_entropy_test_accuracy(params, PC_scenes_training, PC_scenes_test, verbose)
        return objective

    best_parameters, best_values, experiment, model = optimize(
        parameters=[
            {
                "name": "rmin",
                "type": "range",
                "value_type": "float",
                "bounds": [0.5, 4.0],
            },
            {
                "name": "rmax",
                "type": "range",
                "value_type": "float",
                "bounds": [2.0, 10.0],
            },
            {
                "name": "log_epsilon",
                "type": "range",
                "value_type": "float",
                "bounds": [-22, -16],  # [10e-10, 10e-7]
            },
            {
                "name": "alpha",
                "type": "range",
                "value_type": "float",
                "bounds": [2.0, 10.0],
            },
            {
                "name": "E_reject",
                "type": "range",
                "value_type": "float",
                "bounds": [0.10, 0.30],
            },
        ],
        parameter_constraints=["rmax - rmin >= 0"],
        evaluation_function=evaluation_function_wrapper,
        minimize=False,
        total_trials=total_trials,
    )
    print("\n\nOptimization done! Here is the result.")
    rounded_parameters = {param: round(value, 4) for param, value in best_parameters.items()}
    print(f"Best estimated parameters: {rounded_parameters}")
    print(f"Best test accuracy:        {best_values[0]['objective']}")