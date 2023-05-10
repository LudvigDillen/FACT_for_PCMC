from ax import optimize

from classifiers.differential_entropy import differential_entropy_test_accuracy
from utils.data_handling import read_nuscenes_data, gather_data


def optimize_with_ax():
    PC_scenes = read_nuscenes_data()
    PC_scenes_training, PC_scenes_test = gather_data(PC_scenes, samples_training=5, samples_test=5)

    def evaluation_function_wrapper(params):
        return differential_entropy_test_accuracy(params, PC_scenes_training, PC_scenes_test)

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
                "bounds": [-23.025850929940457, -16.11809565095832],  # [10e-10, 10e-7]
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
                "bounds": [0.05, 0.30],
            },
        ],
        evaluation_function=evaluation_function_wrapper,
        minimize=False,
    )
    print(best_parameters, best_values, experiment, model)
