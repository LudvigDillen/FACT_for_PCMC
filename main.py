from utils.other import start_debug
from classifiers.PointTransformers.train_cls import main as fact


if __name__ == "__main__":
    # start_debug()
    fact()
    # optimize_with_ax(samples_training=100, samples_test=40, verbose=True, total_trials=40)
