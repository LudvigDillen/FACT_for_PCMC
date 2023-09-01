from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


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


def plot_accuracies(train_accuracies, val_accuracies, plot_train_acc=False, model_identifier='none'):
    plt.figure(figsize=(10, 5))
    if plot_train_acc:
        plt.plot(range(1, len(train_accuracies) + 1), 100*np.array(train_accuracies), label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), 100*np.array(val_accuracies), label='Validation Accuracy')
    if plot_train_acc:
        plt.title('Training and Validation Accuracy')
    else:
        plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid(True)

    # Ensure that the x-axis only uses integer values
    result = len(val_accuracies) / 20
    tick_step = np.ceil(result / 10) * 10

    plt.xticks(ticks=np.arange(tick_step, len(val_accuracies) + 1, step=tick_step))

    # TODO: Change directory to something chosen in the .cls file
    # Define the directory and filename
    directory = '/home/luddi824/thesis/PCAC/images/classification/PointTransformer'

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime('%Y%m%d_%H%M%S')

    filename = f'accuracy_plot_{time_string}_{model_identifier}'

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

    # TODO: Change directory to something chosen in the .cls file
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
            if "Train Instance Accuracy (regular data)" in line:
                train_accuracies.append(float(words[-1]))
            elif "Vali Instance Accuracy" in line:
                test_accuracies.append(float(words[-1]))

    # Convert lists to numpy arrays
    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    return train_accuracies, test_accuracies


def text_color(bg_color):
    """Return 'black' or 'white' depending on the perceived brightness of bg_color."""
    brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return 'black' if brightness > 0.5 else 'white'


def store_confusion_matrix(y_pred, y_true, N_classes, logger, model_identifier):
    # Create a list of all expected classes
    classes = list(range(N_classes))

    # Calculate the confusion matrix, ensuring it has the expected shape
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    logger.info(f'Confusion matrix {model_identifier}\n {cm}')

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)

    # Setting x and y axis labels
    class_labels = ['Class {}'.format(i) for i in range(N_classes)]
    ax.set_xticks(np.arange(N_classes))
    ax.set_yticks(np.arange(N_classes))
    fontsize = min(int(60/N_classes), 11)
    ax.set_xticklabels(class_labels, fontsize=fontsize)
    ax.set_yticklabels(class_labels, fontsize=fontsize)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(False)  # Hide the grid lines

    # Display the counts on the matrix
    for i in range(N_classes):
        for j in range(N_classes):
            cell_color = cax.to_rgba(cm[i, j])[:3]
            plt.text(j, i, cm[i, j], ha='center', va='center', color=text_color(cell_color))

    plt.tight_layout()

    # TODO: Change directory to something chosen in the .cls file
    # Define the directory and filename
    directory = '/home/luddi824/thesis/PCAC/images/classification/PointTransformer'

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime('%Y%m%d_%H%M%S')

    filename = f'confusion_matrix_{time_string}_{model_identifier}'

    # Save the figure to an .eps file
    fig.savefig(f'{directory}/{filename}.eps', format='eps')
    # Save the figure to an .eps file
    fig.savefig(f'{directory}/{filename}.jpg', format='jpg')

    plt.close()  # Close the figure


if __name__ == "__main__":
    filename = input("Enter the path to the file: ")
    train_accuracies, test_accuracies = extract_accuracies(filename)
    print("Train Accuracies:", train_accuracies)
    print("Test Accuracies:", test_accuracies)
    plot_accuracies(train_accuracies, test_accuracies, plot_train_acc=True)
