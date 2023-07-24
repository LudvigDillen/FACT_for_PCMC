from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def model_plot(model, X, y, title):
    parm = {}
    b = []
    for name, param in model.named_parameters():
        parm[name] = param.detach().numpy()

    w = parm['linear.weight'][0]
    b = parm['linear.bias'][0]
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
    u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
    plt.plot(u, (0.5-b-w[0]*u)/w[1])
    plt.xlim(X[:, 0].min()-0.1, X[:, 0].max()+0.1)
    plt.ylim(X[:, 1].min()-0.1, X[:, 1].max()+0.1)
    # Normally you can just add the argument fontweight='bold' but it does not work with latex
    plt.xlabel(r'x_1', fontsize=16, fontweight='bold')
    plt.ylabel(r'x_2', fontsize=16, fontweight='bold')
    plt.title(title)
    plt.show()


def plot_accuracies(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), 100*np.array(train_accuracies), label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), 100*np.array(val_accuracies), label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid(True)

    # Ensure that the x-axis only uses integer values
    result = len(train_accuracies) / 20
    tick_step = np.ceil(result / 10) * 10

    plt.xticks(ticks=np.arange(tick_step, len(train_accuracies) + 1, step=tick_step))

    # Define the directory and filename
    directory = '/home/luddi824/thesis/PCAC/images/classification/PointTransformer'

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime('%Y%m%d_%H%M%S')

    filename = f'accuracy_plot_{time_string}'

    # Save the figure as .jpg
    plt.savefig(f'{directory}/{filename}.jpg', format='jpg')

    # Save the figure as .eps
    plt.savefig(f'{directory}/{filename}.eps', format='eps')

    plt.close()  # Close the figure


def plot_accuracies_ablation(all_train_accuracies, all_val_accuracies, true_keys, plot_train=True):
    plt.figure(figsize=(10, 5))

    n_features = len(true_keys)
    legend_names = get_legend_names(true_keys)
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'x', 'p', '*', '+']

    for idx in range(n_features):
        linestyle = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]

        if plot_train:
            # Plotting training accuracies
            plt.plot(range(1, all_train_accuracies.shape[1] + 1), 100 * all_train_accuracies[idx, :],
                     label=f'Train {legend_names[idx]}', linestyle=linestyle,
                     marker=marker, markevery=1)
            val_str = f'Val {legend_names[idx]}'
        else:
            val_str = f'{legend_names[idx]}'
            # Plotting validation accuracies
        plt.plot(range(1, all_val_accuracies.shape[1] + 1), 100 * all_val_accuracies[idx, :],
                 label=val_str, linestyle=linestyle, marker=marker, markevery=1,
                 alpha=0.7)  # reduced opacity for validation
    if plot_train:
        plt.title('Training and Validation Accuracy')
    else:
        plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid(True)

    # Ensure that the x-axis only uses integer values
    result = all_train_accuracies.shape[1] / 20
    tick_step = np.ceil(result / 10) * 10
    plt.xticks(ticks=np.arange(tick_step, all_train_accuracies.shape[1] + 1, step=tick_step))

    # Define the directory and filename
    directory = '/home/luddi824/thesis/PCAC/images/classification/PointTransformer'

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime('%Y%m%d_%H%M%S')

    filename = f'accuracy_plot_{time_string}'

    # Save the figure as .jpg
    plt.savefig(f'{directory}/{filename}.jpg', format='jpg')

    # Save the figure as .eps
    plt.savefig(f'{directory}/{filename}.eps', format='eps')

    plt.close()  # Close the figure


def get_legend_names(keys):
    legend_names = []
    for key in keys:
        legend_names.append(key.split('_')[1])
    return legend_names


def extract_accuracies(filename):
    # Lists to store the accuracies
    train_accuracies = []
    test_accuracies = []

    # Open and read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line into words
            words = line.split()

            # Check if the line contains the accuracies
            if "Train Instance Accuracy" in line:
                train_accuracies.append(float(words[-1]))
            elif "Test Instance Accuracy" in line:
                test_accuracies.append(float(words[-1]))

    # Convert lists to numpy arrays
    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    return train_accuracies, test_accuracies


if __name__ == "__main__":
    filename = input("Enter the path to the file: ")
    train_accuracies, test_accuracies = extract_accuracies(filename)
    print("Train Accuracies:", train_accuracies)
    print("Test Accuracies:", test_accuracies)
    plot_accuracies(train_accuracies, test_accuracies)
