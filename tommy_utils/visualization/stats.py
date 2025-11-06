"""
Statistical plotting utilities.

This module provides functions for creating statistical visualizations including
scatter-boxplots, scatter-barplots, KDE-boxplots, and correlation matrices.
"""

from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
import seaborn as sns
import numpy as np


def chunkwise(t, size=2):
    """
    Split an iterable into chunks of specified size.

    Parameters
    ----------
    t : iterable
        Input iterable to chunk
    size : int, default=2
        Size of each chunk

    Returns
    -------
    zip
        Iterator of tuples with size elements each
    """
    it = iter(t)
    return zip(*[it]*size)


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
        palette = sns.cubehelix_palette(start=0.5, rot=-.5, dark=0.5, light=0.9)[::-1]

    if ax is None:
        fig, ax = plt.subplots()

    sns.boxplot(x=x, y=y, hue=group, data=df, saturation=1, showfliers=False, order=order, hue_order=hue_order,
            width=0.8, linewidth=2, palette=palette, ax=ax,
                medianprops={'color': 'black'}, whiskerprops={'color': 'black'}, capprops={'visible': False})

    box_colors = [c for i in range(n_groups) for c in palette]
    box_patches = [p for p in ax.patches if isinstance(p, PathPatch)]

    for patch, color in zip(box_patches, box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')  # Set the edge color to black

    # Add hatches to the second box in each pair
    hatches = ['', '///']

    for i, patch in enumerate(box_patches):
        if i >= n_items:
            patch.set_hatch(hatches[1])

    sns.stripplot(x=x, y=y, hue=group, data=df, marker='o', color='0.9', alpha=0.4,
                  edgecolor='0.1', linewidth=0.15, dodge=True, palette=palette, ax=ax,
                  zorder=10)

    dot_colors = sum([[c]*2 for c in box_colors], [])

    for collection, color in zip(ax.collections, dot_colors):
        collection.set_facecolor(color)

    if group is not None:
        ax.get_legend().remove()

        for ax1, ax2 in chunkwise(ax.collections, size=2):
            for (x0, y0), (x1, y1) in zip(ax1.get_offsets(), ax2.get_offsets()):
                ax.plot([x0, x1], [y0, y1], color='black', alpha=0.15, linewidth=0.5)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2])

    if not use_legend:
        plt.legend([],[], frameon=False)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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
        palette = sns.cubehelix_palette(start=0.5, rot=-.5, dark=0.5, light=0.9)[::-1]

    ax = sns.violinplot(y=y, x=x, data=df,
                    palette=palette, dodge=False,
                    scale="width", inner=None, cut=cut)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        violin.set_alpha(alpha)
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds

        if direction == 'horizontal':
            violin.set_clip_path(plt.Rectangle((x0, y0), width, height/2, transform=ax.transData))
            dot_offset = np.asarray([0, 0.15])
        elif direction == 'vertical':
            violin.set_clip_path(plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
            dot_offset = np.asarray([0.15, 0])

    # plot box plots
    sns.boxplot(y=y, x=x, data=df, saturation=1, showfliers=False,
            width=0.3, linewidth=1.5, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)

    old_len_collections = len(ax.collections)
    sns.stripplot(y=y, x=x, data=df, marker='o', color='0.9', edgecolor='0.1', alpha=0.25, linewidth=0.5, ax=ax)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + dot_offset)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax


def scatter_barplot(df, x, y, group=None, palette='RdBu_r', ax=None, order=None, ci=95, use_legend=False, reverse_hatches=False, plot_points=True):
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
        palette = sns.cubehelix_palette(start=0.5, rot=-.5, dark=0.5, light=0.9)[::-1]

    if ax is None:
        fig, ax = plt.subplots()

    # Plot bars with error bars - set errcolor and capsize
    bar_plot = sns.barplot(x=x, y=y, hue=group, data=df, palette=palette,
                          order=order, hue_order=hue_order, ci=ci, ax=ax,
                          edgecolor='black', saturation=1, linewidth=1.5, zorder=2,
                          errcolor='black', errwidth=1.5)  # Add error bar styling

    # Plot individual points FIRST (lower zorder)
    if plot_points:
        dodge = True if group is not None else False
        sns.stripplot(x=x, y=y, hue=group, data=df,
                    marker='o', color='0.9', alpha=0.4, edgecolor='0.1',
                    linewidth=0.3, dodge=dodge, palette=palette, ax=ax,
                    order=order, zorder=3)  # Lower zorder than error bars

    # Get error bar lines and set them to higher zorder
    # Error bars are Line2D objects in ax.lines
    for line in ax.lines:
        # Check if this is an error bar line (they're usually the vertical lines)
        if line.get_linestyle() == '-':
            line.set_color('black')
            line.set_linewidth(1.5)
            line.set_zorder(20)  # Put error bars on top

    # Set colors and hatches
    if group is not None:
        box_colors = [c for i in range(n_groups) for c in palette]

        # Get all patches and set their colors
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
                    # Create a new patch with the same dimensions but only hatch
                    x_pos, y_pos = patch.get_x(), patch.get_y()
                    width, height = patch.get_width(), patch.get_height()

                    hatch_patch = plt.Rectangle((x_pos, y_pos), width, height,
                                            facecolor='none',
                                            hatch=hatches[1],
                                            edgecolor=box_colors[color_idx],
                                            linewidth=0,
                                            zorder=1)  # Behind bars but above background
                    ax.add_patch(hatch_patch)

                    # Keep the original patch visible with white/transparent face
                    patch.set_facecolor('none')
                    patch.set_alpha(1)  # Semi-transparent instead of invisible
                else:
                    patch.set_hatch(hatches[1])

    # Set dot colors
    if group is not None:
        dot_colors = sum([[c]*2 for c in box_colors], [])
        for collection, color in zip(ax.collections, dot_colors):
            collection.set_facecolor(color)

    if group is not None:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Connect points between groups
        for ax1, ax2 in chunkwise(ax.collections, size=2):
            for (x0, y0), (x1, y1) in zip(ax1.get_offsets(), ax2.get_offsets()):
                ax.plot([x0, x1], [y0, y1], color='black', alpha=0.1,
                       linewidth=0.8, zorder=4)  # Below error bars

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2])

    if not use_legend:
        plt.legend([],[], frameon=False)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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
        plt.subplot(1, n_values, i+1)

        sns.heatmap(matrices[idxs][i], square=True, cbar_kws={"shrink": .25}, cmap='RdBu_r', vmax=vmax, vmin=-vmax)

        plt.title(f'{labels[idxs[i]]}')

    plt.suptitle(title, y=0.75)
    plt.tight_layout()

    plt.savefig(out_fn, bbox_inches='tight')
    plt.close('all')
