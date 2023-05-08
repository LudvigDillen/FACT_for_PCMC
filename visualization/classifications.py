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
