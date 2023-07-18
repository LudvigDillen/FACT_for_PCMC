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
    plt.xticks(ticks=np.arange(1, len(train_accuracies) + 1, step=1))

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

