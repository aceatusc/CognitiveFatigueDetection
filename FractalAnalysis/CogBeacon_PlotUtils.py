

import os
import numpy as np # type: ignore
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mean_Dq(Dq_list, q_values, imagename, subject_labels=None, title='Mean Dq Plot'):
    """
    Plots mean Dq values for multiple subjects dynamically.

    Input:
    - Dq_list: list of 2D arrays, each containing Dq values for a subject.
    - q_values: array or list of q values for the x-axis.
    - subject_labels: list of labels for each subject. Defaults to 'Mean Dq0', 'Mean Dq1', etc.
    - title: title for the plot.
    - imagename: name of the plot to save
    """

    subject_means = [np.squeeze(np.mean(Dq, axis=0)) for Dq in Dq_list]
    # print("subject_means", subject_means)


    if subject_labels is None:
        subject_labels = [f'Mean Dq{i}' for i in range(len(subject_means))]


    plt.figure(figsize=(8, 5))
    for i, mean_Dq in enumerate(subject_means):
        plt.plot(q_values, mean_Dq, '-', label=subject_labels[i], alpha=0.7)

    # Add labels, title, and legend
    plt.xlabel('q')
    plt.ylabel('Mean Dq Values')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join("cogBeacon_plots/Dq/", imagename), dpi=300)
    # plt.show()




# def plot_dq_windows(Dq,q,overlap, windowsize, filename, fmt = 'ko-',ax=None):
#         """
#         Plot the multifractal spectrum.

#         Parameters
#         ----------
#         figlabel : str
#             Figure title
#         filename : str | None
#             If not None, path used to save the figure
#         """

#         ax = plt.gca() if ax is None else ax
#         CI_Dq, CI_hq = None, None

#         ax.errorbar(q, Dq.reshape(Dq.shape[0]),CI_Dq, CI_hq, fmt)

#         ax.set(xlabel='q', ylabel='D(q)')
#         plt.draw()

#         if filename is not None:
#             filepath = os.path.join("CogBeaconPlots/Dq_all_window/", f"{filename}_plot_overlap_{overlap}_window_{windowsize}.png")
#             plt.savefig(filepath)



def plot_dq_windows_dif(Dq_values, q_values, median_labels, overlap, windowsize, filename, fmt='-o', ax=None):
    """
    Plot the absolute difference in multifractal spectrum (|D(q)_i - D(q)_{i+1}|) for consecutive windows.

    Parameters
    ----------
    Dq_values : list of arrays
        List containing Dq arrays for each window
    q_values : list of arrays
        List containing q arrays for each window
    overlap : bool
        Indicates whether windows overlap or not
    windowsize : int
        Size of the window
    filename : str | None
        Path used to save the figure if provided
    fmt : str
        Format for the plot (default is '-')
    ax : matplotlib axis, optional
        Axis on which to plot. Creates a new one if not provided
    """
    fig, ax = plt.subplots(figsize=(7, 4))  # Create a new figure and axis

    # Use a larger color map with high variability
    colors = plt.cm.tab20(np.linspace(0, 1, len(Dq_values) - 1))

    # Plot the difference in Dq values between each consecutive window
    for i in range(len(Dq_values) - 1):
        dq_diff = np.abs(Dq_values[i].reshape(-1) - Dq_values[i+1].reshape(-1))
        q = q_values[i].reshape(-1)  # Assuming q_values are the same for consecutive windows
        ax.plot(q, dq_diff, fmt, label=f'|Dq{i+1} - Dq{i+2}| (Fatigue: {median_labels[i]})', color=colors[i])

    ax.set(xlabel='q', ylabel='|D(q) Difference|')

    # Move the legend to the side
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    ax.set_title(f"{filename}_window_{windowsize}")

    plt.tight_layout()  # Adjust layout to fit larger legend

    # Save the plot without displaying it
    if filename is not None:
        filepath = os.path.join("CogBeaconPlots/Dq_all_window_dif/", f"{filename}_plot_overlap_{overlap}_window_{windowsize}_diff.png")
        plt.savefig(filepath, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free up memory




def plot_dq_windows(Dq_values, q_values, median_labels, overlap, windowsize, filename, fmt='-o', ax=None):
    """
    Plot the multifractal spectrum for each window.

    Parameters
    ----------
    Dq_values : list of arrays
        List containing Dq arrays for each window
    q_values : list of arrays
        List containing q arrays for each window
    median_labels : list
        List of median labels for each window
    overlap : bool
        Indicates whether windows overlap or not
    windowsize : int
        Size of the window
    filename : str | None
        Path used to save the figure if provided
    fmt : str
        Format for the plot (default is '-o')
    ax : matplotlib axis, optional
        Axis on which to plot. Creates a new one if not provided
    """
    fig, ax = plt.subplots(figsize=(7, 4))  # Create a new figure and axis

    # Define a list of colors for each plot
    colors = plt.cm.tab20(np.linspace(0, 1, len(Dq_values)))

    # Plot Dq vs q for each window with explicit index
    for i, (dq, q) in enumerate(zip(Dq_values, q_values)):
        dq = dq.reshape(-1)
        q = q.reshape(-1)
        ax.plot(q, dq, fmt, color=colors[i], label=f'Dq{i + 1} (Fatigue: {median_labels[i]})')

    ax.set(xlabel='q', ylabel='D(q)')
    ax.set_title(f"{filename}_window_{windowsize}")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')


    # Save the plot without displaying it
    if filename is not None:
        filepath = os.path.join("CogBeaconPlots/Dq_all_window/", f"{filename}_plot_overlap_{overlap}_window_{windowsize}.png")
        plt.savefig(filepath, bbox_inches='tight')

    plt.close(fig)  # Close the figure to free up memory



# def plot_dq_windows_dif(Dq_values, q_values, overlap, windowsize, filename, fmt='-', ax=None):
#     """
#     Plot the multifractal spectrum with error bars and distinct colors for each window.

#     Parameters
#     ----------
#     Dq_values : list of arrays
#         List containing Dq arrays for each window
#     q_values : list of arrays
#         List containing q arrays for each window
#     overlap : bool
#         Indicates whether windows overlap or not
#     windowsize : int
#         Size of the window
#     filename : str | None
#         Path used to save the figure if provided
#     fmt : str
#         Format for the plot (default is 'ko-')
#     ax : matplotlib axis, optional
#         Axis on which to plot. Creates a new one if not provided
#     """
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(10, 8)) if ax is None else ax

#     # Use a color map for varied colors for each plot
#     colors = plt.cm.tab10(np.linspace(0, 1, len(Dq_values)))

#     # Plot each Dq curve with distinct color and 'ko-' marker style
#     for i, (dq, q) in enumerate(zip(Dq_values, q_values)):
#         dq = dq.reshape(-1)
#         q = q.reshape(-1)
#         CI_Dq, CI_hq = None, None  # Placeholder, adjust if needed for actual error values
#         ax.errorbar(q, dq, yerr=CI_Dq, xerr=CI_hq, fmt=fmt, color=colors[i], label=f'Window {i+1}')

#     ax.set(xlabel='q', ylabel='D(q)')
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
#     plt.draw()

#     if filename is not None:
#         filepath = os.path.join("CogBeaconPlots/Dq_all_window_dif/", f"{filename}_plot_overlap_{overlap}_window_{windowsize}.png")
#         plt.savefig(filepath, bbox_inches='tight')




def plot_mfda_mean_per_transition_combined(avg_results, figlabel='MFDA Features', filename_prefix=None, **plot_kwargs):
    """
    Plot the mean values of MFDA features for each transition in a 2D grid of subplots, showing before and after in each plot.

    Parameters
    ----------
    avg_results : dict
        A dictionary containing the averaged mfda features before and after transitions.
        It should contain keys: 'avg_mfda_before_windows' and 'avg_mfda_after_windows'.
    figlabel : str
        Figure title prefix
    filename_prefix : str | None
        If not None, prefix used to save the figures with transition labels
    plot_kwargs : dict
        Additional keyword arguments for the plot
    """

    transitions = list(avg_results['avg_mfda_before_windows'].keys())
    num_transitions = len(transitions)

    # Define the number of rows and columns for the grid
    cols = 2  # Number of columns in the grid (before and after)
    rows = (num_transitions + cols - 1) // cols  # Calculate number of rows needed

    # Create a new figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    colors = plt.cm.viridis(np.linspace(0, 1, num_transitions * 2))

    for idx, transition in enumerate(transitions):
        before_avg = avg_results['avg_mfda_before_windows'][transition]
        after_avg = avg_results['avg_mfda_after_windows'][transition]

        ax = axes[idx]  # Select the current axis

        # Plot before averages
        if before_avg is not None:
            ax.plot(avg_results['q'], before_avg.reshape(before_avg.shape[0]), 'ko-', markersize=3, color='blue', label=f'Before {transition}', **plot_kwargs)

        # Plot after averages
        if after_avg is not None:
            ax.plot(avg_results['q'], after_avg.reshape(after_avg.shape[0]), 'ko-', markersize=3, color='red', label=f'After {transition}', **plot_kwargs)

        # Adding plot details
        ax.set(xlabel='q', ylabel='Dq', ylim=(-0.5, 1.5))
        ax.set_title(f'{figlabel}: {transition}')
        ax.legend()
        sns.despine()

    # Hide any unused subplots
    for ax in axes[num_transitions:]:
        ax.set_visible(False)

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    
    # Save the figure if a filename prefix is provided
    if filename_prefix is not None:
        plt.savefig(f'{filename_prefix}_transitions.png')

    plt.show()  # Show the plot
