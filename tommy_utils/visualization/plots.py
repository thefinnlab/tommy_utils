"""
Statistical plotting utilities.

This module provides functions for creating statistical visualizations including
scatter-boxplots, scatter-barplots, KDE-boxplots, and correlation matrices.
"""

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
import seaborn as sns
import numpy as np

from ._helpers import (
    get_default_palette,
    remove_top_right_spines,
    setup_group_legend,
    set_collection_colors,
    chunkwise,
    connect_grouped_points
)


def scatter_boxplot(df, x, y, group=None, palette='RdBu_r', order=None, ax=None, use_legend=False):
    """
    Create a box plot with individual data points and connecting lines between groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    x : str
        Column name for x-axis categories
    y : str
        Column name for y-axis values
    group : str, optional
        Column name for grouping
    palette : str or list, default='RdBu_r'
        Color palette for the plot
    order : list, optional
        Order of categories on x-axis
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    use_legend : bool, default=False
        Whether to show the legend

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    n_items = len(np.unique(df[x]))
    n_groups = len(np.unique(df[group]))
    hue_order = np.unique(df[group])

    if not palette:
        palette = get_default_palette()

    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(
        x=x, y=y, hue=group, data=df, saturation=1, showfliers=False,
        order=order, hue_order=hue_order, width=0.8, linewidth=2,
        palette=palette, ax=ax, medianprops={'color': 'black'},
        whiskerprops={'color': 'black'}, capprops={'visible': False}
    )

    box_colors = [c for i in range(n_groups) for c in palette]
    box_patches = [p for p in ax.patches if isinstance(p, PathPatch)]

    for patch, color in zip(box_patches, box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    # Add hatches to the second box in each pair
    hatches = ['', '///']
    for i, patch in enumerate(box_patches):
        if i >= n_items:
            patch.set_hatch(hatches[1])

    sns.stripplot(
        x=x, y=y, hue=group, data=df, marker='o', color='0.9', alpha=0.4,
        edgecolor='0.1', linewidth=0.15, dodge=True, palette=palette,
        ax=ax, zorder=10
    )

    set_collection_colors(ax, box_colors)

    if group is not None:
        ax.get_legend().remove()
        connect_grouped_points(ax, alpha=0.15, linewidth=0.5)
        setup_group_legend(ax, n_labels=2)

    if not use_legend:
        plt.legend([], [], frameon=False)

    remove_top_right_spines(ax)

    return ax


def kde_boxplot(df, x, y, direction='horizontal', palette=None, alpha=0.5, cut=2):
    """
    Create a combined KDE (violin) and box plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    x : str
        Column name for x-axis categories
    y : str
        Column name for y-axis values
    direction : str, default='horizontal'
        Direction to clip the violin plot ('horizontal' or 'vertical')
    palette : list, optional
        Color palette for the plot
    alpha : float, default=0.5
        Transparency of the violin plot
    cut : float, default=2
        Distance beyond data for KDE estimation

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if not palette:
        palette = get_default_palette()

    ax = sns.violinplot(
        y=y, x=x, data=df, palette=palette, dodge=False,
        scale="width", inner=None, cut=cut
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        violin.set_alpha(alpha)
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds

        if direction == 'horizontal':
            violin.set_clip_path(
                plt.Rectangle((x0, y0), width, height / 2, transform=ax.transData)
            )
            dot_offset = np.asarray([0, 0.15])
        elif direction == 'vertical':
            violin.set_clip_path(
                plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
            )
            dot_offset = np.asarray([0.15, 0])

    # Plot box plots
    sns.boxplot(
        y=y, x=x, data=df, saturation=1, showfliers=False,
        width=0.3, linewidth=1.5, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax
    )

    old_len_collections = len(ax.collections)
    sns.stripplot(
        y=y, x=x, data=df, marker='o', color='0.9', edgecolor='0.1',
        alpha=0.25, linewidth=0.5, ax=ax
    )
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + dot_offset)

    remove_top_right_spines(ax)

    return ax


def scatter_barplot(df, x, y, group=None, palette='RdBu_r', ax=None, order=None,
                    ci=95, use_legend=False, reverse_hatches=False, plot_points=True):
    """
    Create a bar plot with individual data points.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    x : str
        Column name for x-axis categories
    y : str
        Column name for y-axis values
    group : str, optional
        Column name for grouping
    palette : str, default='RdBu_r'
        Color palette for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    order : list, optional
        Order of categories on x-axis
    ci : int, default=95
        Confidence interval for error bars
    use_legend : bool, default=False
        Whether to show the legend
    reverse_hatches : bool, default=False
        Whether to reverse the hatch pattern styling
    plot_points : bool, default=True
        Whether to plot individual data points

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot
    """
    if group is not None:
        hue_order = np.unique(df[group])
        n_groups = len(hue_order)
        n_items = len(np.unique(df[x]))
    else:
        hue_order = None
        n_groups = 1
        n_items = len(np.unique(df[x]))

    if not palette:
        palette = get_default_palette()

    if ax is None:
        fig, ax = plt.subplots()

    # Plot bars with error bars
    bar_plot = sns.barplot(
        x=x, y=y, hue=group, data=df, palette=palette,
        order=order, hue_order=hue_order, ci=ci, ax=ax,
        edgecolor='black', saturation=1, linewidth=1.5, zorder=2,
        errcolor='black', errwidth=1.5
    )

    # Plot individual points
    if plot_points:
        dodge = True if group is not None else False
        sns.stripplot(
            x=x, y=y, hue=group, data=df, marker='o', color='0.9',
            alpha=0.4, edgecolor='0.1', linewidth=0.3, dodge=dodge,
            palette=palette, ax=ax, order=order, zorder=3
        )

    # Style error bars
    for line in ax.lines:
        if line.get_linestyle() == '-':
            line.set_color('black')
            line.set_linewidth(1.5)
            line.set_zorder(20)

    # Set colors and hatches
    if group is not None:
        box_colors = [c for i in range(n_groups) for c in palette]

        for i, patch in enumerate(ax.patches):
            color_idx = i % len(box_colors)
            patch.set_facecolor(box_colors[color_idx])
            patch.set_edgecolor('black')

        # Add hatches to the second bar in each pair
        hatches = ['', '///']
        for i, patch in enumerate(ax.patches):
            if i >= n_items:
                color_idx = (i - n_items) % len(box_colors)

                if reverse_hatches:
                    x_pos, y_pos = patch.get_x(), patch.get_y()
                    width, height = patch.get_width(), patch.get_height()

                    hatch_patch = plt.Rectangle(
                        (x_pos, y_pos), width, height,
                        facecolor='none', hatch=hatches[1],
                        edgecolor=box_colors[color_idx],
                        linewidth=0, zorder=1
                    )
                    ax.add_patch(hatch_patch)

                    patch.set_facecolor('none')
                    patch.set_alpha(1)
                else:
                    patch.set_hatch(hatches[1])

    if group is not None:
        set_collection_colors(ax, box_colors)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Connect points between groups
        connect_grouped_points(ax, alpha=0.1, linewidth=0.8, zorder=4)

        setup_group_legend(ax, n_labels=2)

    if not use_legend:
        plt.legend([], [], frameon=False)

    remove_top_right_spines(ax)

    return ax


def plot_regressor_raster(features, times=None, labels=None, cmap='viridis', ax=None,
                          aspect='auto', xlabel='Time (s)', ylabel='Regressor',
                          colorbar=True, vmin=None, vmax=None, **kwargs):
    """
    Create a raster plot of regressors over time.

    Displays multiple regressors as horizontal bands where color intensity
    represents the regressor value at each time point. Useful for visualizing
    encoding model features, stimulus timecourses, or any time-varying signals.

    Parameters
    ----------
    features : np.ndarray or list
        Regressor data. Can be:
        - 2D array of shape (n_timepoints, n_regressors)
        - 2D array of shape (n_regressors, n_timepoints) if transposed
        - List of 1D arrays (will be stacked)
    times : np.ndarray, optional
        Time points in seconds. If None, uses sample indices.
    labels : list of str, optional
        Labels for each regressor (shown on y-axis)
    cmap : str, default='viridis'
        Colormap for the raster plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    aspect : str or float, default='auto'
        Aspect ratio of the plot
    xlabel : str, default='Time (s)'
        Label for x-axis
    ylabel : str, default='Regressor'
        Label for y-axis
    colorbar : bool, default=True
        Whether to show a colorbar
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    **kwargs : dict
        Additional arguments passed to plt.imshow()

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot

    Examples
    --------
    >>> # Plot scene cuts and motion energy over time
    >>> times, scene_cuts = create_scene_cut_features('video.mp4')
    >>> times, motion = create_motion_energy_features(decoder)
    >>> plot_regressor_raster(
    ...     np.hstack([scene_cuts, motion[:, :5]]),  # First 5 motion filters
    ...     times=times,
    ...     labels=['Scene cuts'] + [f'Motion {i}' for i in range(5)]
    ... )

    >>> # Plot multiple feature types
    >>> features = [audio_envelope, pitch_contour, scene_cuts.squeeze()]
    >>> plot_regressor_raster(features, times=times, labels=['Audio', 'Pitch', 'Scenes'])
    """
    # Handle list of arrays
    if isinstance(features, list):
        features = np.column_stack([np.atleast_2d(f).T if f.ndim == 1 else f for f in features])

    # Ensure features is 2D with shape (n_timepoints, n_regressors)
    features = np.atleast_2d(features)
    if features.shape[0] < features.shape[1]:
        # Assume (n_regressors, n_timepoints) -> transpose
        features = features.T

    n_timepoints, n_regressors = features.shape

    # Create time array if not provided
    if times is None:
        times = np.arange(n_timepoints)

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(3, n_regressors * 0.5)))

    # Create extent for proper axis scaling
    # extent = [left, right, bottom, top]
    extent = [times[0], times[-1], n_regressors - 0.5, -0.5]

    # Plot raster
    im = ax.imshow(
        features.T,  # Transpose so regressors are on y-axis
        aspect=aspect,
        cmap=cmap,
        extent=extent,
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
        **kwargs
    )

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set y-tick labels if provided
    if labels is not None:
        ax.set_yticks(np.arange(n_regressors))
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks(np.arange(n_regressors))
        ax.set_yticklabels([f'R{i+1}' for i in range(n_regressors)])

    # Add colorbar
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label='Value')

    remove_top_right_spines(ax)

    return ax


def plot_correlation_matrices(out_fn, title, n_values, matrices, labels, idxs, vmax):
    """
    Plot multiple correlation matrices side by side.

    Parameters
    ----------
    out_fn : str
        Output filename for saving the plot
    title : str
        Overall title for the figure
    n_values : int
        Number of matrices to plot
    matrices : dict or array-like
        Container of correlation matrices
    labels : list
        Labels for each matrix
    idxs : list
        Indices of matrices to plot
    vmax : float
        Maximum value for color scale
    """
    fig = plt.figure()

    for i in range(n_values):
        plt.subplot(1, n_values, i + 1)

        sns.heatmap(
            matrices[idxs][i], square=True, cbar_kws={"shrink": .25},
            cmap='RdBu_r', vmax=vmax, vmin=-vmax
        )

        plt.title(f'{labels[idxs[i]]}')

    plt.suptitle(title, y=0.75)
    plt.tight_layout()

    plt.savefig(out_fn, bbox_inches='tight')
    plt.close('all')
