import numpy as np
import matplotlib.pyplot as plt
import copy
import open3d as o3d


def plot_reg_error_hists(R_errors_init, R_errors_p2p, R_errors_p2l, n_samples,
                         t_errors_init, t_errors_p2p, t_errors_p2l):
    R_labels = ['identity', 'icp-p2p', 'icp-p2l']
    t_labels = ['zero', 'icp-p2p', 'icp-p2l']

    # Bin boundaries
    r_bin_edges = np.linspace(0, 0.020, num=10).tolist() + [np.inf]
    t_bin_edges = np.linspace(0, 0.9, num=10).tolist() + [np.inf]

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, figure size
    R_bar_width = (r_bin_edges[1] - r_bin_edges[0]) / 6  # Divide by the number of datasets plus some spacing
    t_bar_width = (t_bin_edges[1] - t_bin_edges[0]) / 6  # Divide by the number of datasets plus some spacing

    # Histogram for R_errors
    for i, (data, label) in enumerate(zip([R_errors_init, R_errors_p2p, R_errors_p2l], R_labels)):
        ax1.hist(data.reshape(n_samples).cpu(), bins=r_bin_edges, alpha=0.5,
                 label=label, width=R_bar_width, edgecolor='black')

    # # Adjust the bar positions to be side by side
    for i, rect in enumerate(ax1.patches):
        rect.set_x(rect.get_x() + i // 10 * R_bar_width)

    # Histogram for t_errors
    for i, (data, label) in enumerate(zip([t_errors_init, t_errors_p2p, t_errors_p2l], t_labels)):
        ax2.hist(data.reshape(n_samples).cpu(), bins=t_bin_edges, alpha=0.5,
                 label=label, width=t_bar_width, edgecolor='black')

    # # Adjust the bar positions to be side by side
    for i, rect in enumerate(ax2.patches):
        rect.set_x(rect.get_x() + i // 10 * t_bar_width)

    ax1.set_title('Histogram of rotation errors')
    ax1.set_xlabel('Geodesic distance [rad]')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2.set_title('Histogram of translation errors')
    ax2.set_xlabel('Translation error [m]')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_reg_error_over_samples(R_errors_init, R_errors_p2p, R_errors_p2l, n_samples,
                                t_errors_init, t_errors_p2p, t_errors_p2l):
    x_values = range(n_samples)  # This will create a range from 0 to n_samples-1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, optional figure size

    # First subplot for R_errors
    ax1.plot(x_values, R_errors_init.reshape(n_samples).cpu(), label='R_errors_init')
    ax1.plot(x_values, R_errors_p2p.reshape(n_samples).cpu(), label='R_errors_p2p')
    ax1.plot(x_values, R_errors_p2l.reshape(n_samples).cpu(), label='R_errors_p2l')
    ax1.set_title('R_errors')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Magnitude')
    ax1.legend()

    # Second subplot for t_errors (assuming t_errors_init, t_errors_p2p, t_errors_p2l are defined)
    ax2.plot(x_values, t_errors_init.reshape(n_samples).cpu(), label='t_errors_init')
    ax2.plot(x_values, t_errors_p2p.reshape(n_samples).cpu(), label='t_errors_p2p')
    ax2.plot(x_values, t_errors_p2l.reshape(n_samples).cpu(), label='t_errors_p2l')
    ax2.set_title('t_errors')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Magnitude')
    ax2.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# Function to create labels for the x-ticks
def create_labels(bin_edges, type):
    if type == 'r':
        labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-2)]
        labels.append(f">{bin_edges[-2]:.3f}")
    elif type == 't':
        labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-2)]
        labels.append(f">{bin_edges[-2]:.1f}")
    return labels


def plot_reg_error_hist(R_errors, t_errors):
    # Bin boundaries
    r_bin_edges = np.linspace(0, 0.018, num=10).tolist() + [np.inf]
    t_bin_edges = np.linspace(0, 0.9, num=10).tolist() + [np.inf]

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, figure size

    # Histogram for R_errors
    ax1.hist(R_errors, bins=r_bin_edges, alpha=0.5, label='R_error', edgecolor='black')
    ax2.hist(t_errors, bins=t_bin_edges, alpha=0.5, label='t_error', edgecolor='black')
    # Setting custom x-ticks for ax1
    r_labels = create_labels(r_bin_edges, type='r')
    r_midpoints = [(r_bin_edges[1]-r_bin_edges[0])/2 + r_bin_edges[i] for i in range(len(r_bin_edges)-1)]
    ax1.set_xticks(r_midpoints)
    ax1.set_xticklabels(r_labels, fontsize=8)

    # Setting custom x-ticks for ax2
    t_labels = create_labels(t_bin_edges, type='t')
    t_midpoints = [(t_bin_edges[1]-t_bin_edges[0])/2 + t_bin_edges[i] for i in range(len(t_bin_edges)-1)]
    ax2.set_xticks(t_midpoints)
    ax2.set_xticklabels(t_labels, fontsize=8)

    ax1.set_title('Histogram of rotation errors')
    ax1.set_xlabel('Geodesic distance [rad]')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2.set_title('Histogram of translation errors')
    ax2.set_xlabel('Translation error [m]')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# From open3d: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html#Point-to-point-ICP
def draw_registration_result(source, target, transformation, title):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996],
                                      window_name=title)
    return None
