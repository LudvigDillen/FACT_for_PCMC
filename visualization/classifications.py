"""
Lots of this code was helped create by ChatGPT.
Also, Kudos to Denny Loevlie from whom I've took the function model_plot().
See link. (MIT license exists)
https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
https://gist.github.com/loevlie/5044e62aea2ce625b70d6d6d75113d25
"""
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.metrics import confusion_matrix


def model_plot(model, X, y, title, args):
    parm = {}
    b = []
    for name, param in model.named_parameters():
        parm[name] = param.detach().numpy()

    w = parm["linear.weight"][0]
    b = parm["linear.bias"][0]

    for label, color, name in zip(
        [0, 1], ["#1f77b4", "#d62728"], ["Aligned", "Misaligned"]
    ):
        plt.scatter(
            X[y == label, 0],
            X[y == label, 1],
            color=color,
            s=50,  # Size of the markers
            edgecolors="k",  # Edge color of the markers
            linewidth=0.5,  # Line width of the marker edges
            alpha=1.0,  # Transparency
            label=name,
        )

    u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)

    u = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400)
    v = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400)
    U, V = np.meshgrid(u, v)
    Z = w[0] * U + w[1] * V + b
    plt.contourf(
        U, V, Z, levels=[-np.inf, 0, np.inf], colors=["#1f77b4", "#d62728"], alpha=0.25
    )

    plt.xlim(U.min(), U.max())
    plt.ylim(V.min(), V.max())
    plt.xlabel(r"$H_{joint}$", fontsize=16, fontweight="bold")
    plt.ylabel(r"$H_{sep}$", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="upper left")
    plt.title(title, fontsize=18)
    # Define the directory and filename
    directory = args.visualization_folder

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")

    file_name = f"model_plot_{time_string}_{args.model_identifier}"

    # Save the figure as .jpg
    plt.savefig(f"{directory}/{file_name}.jpg", format="jpg")

    # Save the figure as .eps
    plt.savefig(f"{directory}/{file_name}.eps", format="eps")

    plt.close()  # Close the figure


def ordinal_model_plot(model, X, y, title, args):
    # Create a mesh grid for plotting
    u = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400)
    v = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400)
    U, V = np.meshgrid(u, v)

    # Flatten the grid so that we can feed it into the model
    grid = np.vstack([U.ravel(), V.ravel()]).T

    # Predict class probabilities for each point on the grid
    probs = model.predict(grid)

    # Determine the most likely class for each point
    # This will be the index of the highest probability
    # If your model outputs class probabilities, use `probs.argmax(axis=1)`
    # Otherwise, adjust the following line as needed based on your model's output
    Z = probs.argmax(axis=1)

    # Reshape the class predictions to match the grid shape
    Z = Z.reshape(U.shape)

    # Define colors for each class
    class_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']  # Add more colors if needed

    # Create a custom colormap
    cmap = mcolors.ListedColormap(class_colors[:len(np.unique(y))])

    # Plot the decision boundaries
    plt.contourf(U, V, Z, alpha=0.25, levels=np.arange(y.min(), y.max()+2)-0.5,
                 cmap=cmap)

    # Plot the data points
    # for label in np.unique(y):
    # plt.scatter(X[y == label, 0], X[y == label, 1], c=y, cmap=cmap, edgecolor='k',
    #             label=np.unique(y))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')


    # Add labels, title, etc.
    param_names = model.params.index
    feature_names = [name for name in param_names if '/' not in name]

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    #plt.legend()
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")

    # plt.legend(handles=scatter.legend_elements()[0], title="Classes", labels=np.unique(y))


    # Save the plot
    directory = args.visualization_folder
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    file_name = f"model_plot_{time_string}_{args.model_identifier}"
    plt.savefig(f"{directory}/{file_name}.jpg", format="jpg")
    plt.savefig(f"{directory}/{file_name}.eps", format="eps")
    plt.close()


def plot_accuracies(
    train_accuracies,
    val_accuracies,
    args,
    default_tick_step=10,
):
    plt.figure(figsize=(10, 5))
    if args.plot_train_acc:
        plt.plot(
            range(1, len(train_accuracies) + 1),
            100 * np.array(train_accuracies),
            label="Train Accuracy",
        )
    plt.plot(
        range(1, len(val_accuracies) + 1),
        100 * np.array(val_accuracies),
        label="Validation Accuracy",
    )
    if args.plot_train_acc:
        plt.title("Training and Validation Accuracy")
    else:
        plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.legend()
    plt.grid(True)

    # Ensure that the x-axis only uses integer values
    result = len(val_accuracies) / 20
    tick_step = np.ceil(result / default_tick_step) * default_tick_step

    plt.xticks(ticks=np.arange(tick_step, len(val_accuracies) + 1, step=tick_step))

    directory = args.visualization_folder

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")

    file_name = f"accuracy_plot_{time_string}_{args.model_identifier}"

    # Save the figure as .jpg
    plt.savefig(f"{directory}/{file_name}.jpg", format="jpg")

    # Save the figure as .eps
    plt.savefig(f"{directory}/{file_name}.eps", format="eps")

    plt.close()  # Close the figure


def plot_accuracies_ablation(
    all_val_accuracies,
    feature_keys,
    args,
    all_train_accuracies=None,
    default_tick_step=10,
    model_identifier="ablation_all",
):
    plt.figure(figsize=(10, 5))

    n_features = sum(args.feature_filter) - len(args.ablation.remove_keys)
    legend_names = get_legend_names(feature_keys)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x", "p", "*", "+"]

    for idx in range(n_features):
        linestyle = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]

        if all_train_accuracies is not None:
            # Plotting training accuracies
            plt.plot(
                range(1, all_train_accuracies.shape[1] + 1),
                100 * all_train_accuracies[idx, :],
                label=f"Train {legend_names[idx]}",
                linestyle=linestyle,
                marker=marker,
                markevery=20,
            )
            val_str = f"Val {legend_names[idx]}"
        else:
            val_str = f"{legend_names[idx]}"
            # Plotting validation accuracies
        plt.plot(
            range(1, all_val_accuracies.shape[1] + 1),
            100 * all_val_accuracies[idx, :],
            label=val_str,
            linestyle=linestyle,
            marker=marker,
            markevery=20,
            alpha=0.7,
        )  # reduced opacity for validation
    if all_train_accuracies is not None:
        plt.title("Training and Validation Accuracy")
    else:
        plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.legend()
    plt.grid(True)

    # Ensure that the x-axis only uses integer values
    result = all_val_accuracies.shape[1] / 20
    tick_step = np.ceil(result / default_tick_step) * default_tick_step
    plt.xticks(
        ticks=np.arange(tick_step, all_val_accuracies.shape[1] + 1, step=tick_step)
    )

    directory = args.visualization_folder

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")

    file_name = f"accuracy_plot_{time_string}_{model_identifier}"

    # Save the figure as .jpg
    plt.savefig(f"{directory}/{file_name}.jpg", format="jpg")

    # Save the figure as .eps
    plt.savefig(f"{directory}/{file_name}.eps", format="eps")

    plt.close()  # Close the figure


def get_legend_names(keys):
    legend_names = []
    for key in keys:
        legend_names.append(key.split("_")[1])
    return legend_names


def extract_accuracies(file_name):
    # Lists to store the accuracies
    train_accuracies = []
    test_accuracies = []

    # Open and read the file
    with open(file_name, "r") as file:
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


def text_color(background):
    # A simple heuristic to determine text color based on background intensity.
    return 'white' if sum(background[:3]) < 1.5 else 'black'


def store_confusion_matrix(y_pred, y_true, N_classes, logger, args, accumulate=False):
    # Create a list of all expected classes
    classes = list(range(N_classes))

    # Calculate the confusion matrix, ensuring it has the expected shape
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    logger.info(f"Confusion matrix {args.model_identifier}\n {cm}")

    n_ticks = N_classes + int(accumulate)
    # Extend the confusion matrix with an extra row and column for the sums if required
    extended_cm = np.zeros((n_ticks, n_ticks), dtype=int)
    extended_cm[:N_classes, :N_classes] = cm
    if accumulate:
        row_sums = cm.sum(axis=1)
        extended_cm[:N_classes, N_classes] = row_sums
        col_sums = cm.sum(axis=0)
        extended_cm[N_classes, :N_classes] = col_sums
    cm = extended_cm

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion matrix of the classifier")

    # Draw a thick line to separate the sums
    if accumulate:
        ax.axhline(y=N_classes - 0.5, color='black', linewidth=2)
        ax.axvline(x=N_classes - 0.5, color='black', linewidth=2)

    # Adjust color bar position
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Setting x and y axis labels
    class_labels = ["Class {}".format(i) if i < N_classes else "Sum" for i in range(n_ticks)]
    ax.set_xticks(np.arange(n_ticks))
    ax.set_yticks(np.arange(n_ticks))
    fontsize = min(int(60 / N_classes), 11)
    ax.set_xticklabels(class_labels, fontsize=fontsize, rotation=90)
    ax.set_yticklabels(class_labels, fontsize=fontsize)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.grid(False)  # Hide the grid lines

    # Display the counts on the matrix
    for i in range(n_ticks):
        for j in range(n_ticks):
            if i < N_classes or j < N_classes:
                cell_color = cax.to_rgba(cm[i, j])[:3]
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color=text_color(cell_color)
                )

    plt.tight_layout()

    directory = args.visualization_folder

    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")

    file_name = f"confusion_matrix_{time_string}_{args.model_identifier}"

    # Save the figure to an .eps file
    fig.savefig(f"{directory}/{file_name}.eps", format="eps", bbox_inches='tight')
    # Save the figure to an .jpg file
    fig.savefig(f"{directory}/{file_name}.jpg", format="jpg", bbox_inches='tight')

    plt.close()  # Close the figure


if __name__ == "__main__":
    filename = input("Enter the path to the file: ")
    train_accuracies, val_accuracies = extract_accuracies(filename)
    print("Train Accuracies:", train_accuracies)
    print("Val Accuracies:", val_accuracies)
    # plot_accuracies(
    #     train_accuracies,
    #     val_accuracies,
    #     args,
    #     plot_train_acc=False,
    #     default_tick_step=10,
    # )
