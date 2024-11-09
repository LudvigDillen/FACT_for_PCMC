import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap


# Analyze the relationship between the predictions, fitness, rmse, and ground truths
def analyze_relationship(preds, fitness, rmse, gts, args):
    # Convert inputs to a DataFrame
    data = {
        'Preds': preds.squeeze(),
        'Fitness': fitness.squeeze(),
        'RMSE': rmse.squeeze(),
        'GT': gts.squeeze()
    }
    df = pd.DataFrame(data)

    # Correlation matrix
    corr_matrix = df.corr()
    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['red', 'white', 'blue'], N=256)

    # Heatmap of the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1,
                annot_kws={"size": 15})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    file_name = f"residuals_conf_{time_string}_{args.model_identifier}"
    directory = args.visualization_folder
    # Save the figure to an .eps file
    plt.savefig(f"{directory}/{file_name}.eps", format="eps", bbox_inches='tight')
    # Save the figure to an .jpg file
    plt.savefig(f"{directory}/{file_name}.jpg", format="jpg", bbox_inches='tight')
    plt.close()  # Close the heatmap figure

    print("Correlation Matrix:")
    print(corr_matrix)

    # Scatter plots
    pairplot = sns.pairplot(df)
    # Customize the labels in the pairplot
    for ax in pairplot.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust tick label size
    pairplot.fig.tight_layout()  # Adjust layout
    pairplot.fig.subplots_adjust(top=0.95)  # Adjust title position

    file_name = f"residual_comparison_{time_string}_{args.model_identifier}"
    # Save the figure to an .eps file
    plt.savefig(f"{directory}/{file_name}.eps", format="eps", bbox_inches='tight')
    # Save the figure to an .jpg file
    plt.savefig(f"{directory}/{file_name}.jpg", format="jpg", bbox_inches='tight')
    plt.close()  # Close the figure
