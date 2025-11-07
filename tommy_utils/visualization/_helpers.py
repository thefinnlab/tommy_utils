"""
Internal helper functions for visualization module.

This module contains shared utility functions used across the visualization
subpackage to reduce code duplication.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from neuromaps.datasets import fetch_fslr, fetch_fsaverage, fetch_civet
from neuromaps.transforms import mni152_to_fslr, mni152_to_fsaverage, mni152_to_civet


# Valid mesh densities for each surface type
VALID_DENSITIES = {
    'fsaverage': ['3k', '10k', '41k', '164k'],
    'fslr': ['4k', '8k', '32k', '164k'],
    'civet': ['41k', '164k'],
}

# Transform functions for volume to surface conversion
_TRANSFORM_FUNCS = {
    'fsaverage': (mni152_to_fsaverage, 'fsavg_density'),
    'fslr': (mni152_to_fslr, 'fslr_density'),
    'civet': (mni152_to_civet, 'civet_density'),
}


def get_default_palette():
    """
    Get default cubehelix palette for statistical plots.

    Returns
    -------
    list
        List of RGB color tuples
    """
    return sns.cubehelix_palette(start=0.5, rot=-.5, dark=0.5, light=0.9)[::-1]


def remove_top_right_spines(ax):
    """
    Remove right and top spines from axes for cleaner appearance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def save_and_close_figure(fig_or_plt, out_fn, transparent=False, dpi=300):
    """
    Save figure and close all plots.

    Parameters
    ----------
    fig_or_plt : matplotlib.figure.Figure or module
        Figure object or matplotlib.pyplot module
    out_fn : str or None
        Output filename. If None, figure is not saved
    transparent : bool, default=False
        Whether to save with transparent background
    dpi : int, default=300
        Resolution for saved figure
    """
    if out_fn:
        if hasattr(fig_or_plt, 'savefig'):
            fig_or_plt.savefig(
                out_fn,
                bbox_inches='tight',
                transparent=transparent,
                dpi=dpi
            )
        else:
            plt.savefig(
                out_fn,
                bbox_inches='tight',
                transparent=transparent,
                dpi=dpi
            )
        plt.close('all')


def fetch_surface(surf_type, density):
    """
    Fetch surface mesh based on type and density.

    Parameters
    ----------
    surf_type : str
        Type of surface ('fsaverage', 'fslr', or 'civet')
    density : str
        Target mesh density

    Returns
    -------
    dict
        Surface dictionary from neuromaps containing mesh data

    Raises
    ------
    ValueError
        If surf_type is not recognized
    AssertionError
        If density is not valid for the surface type
    """
    if surf_type not in VALID_DENSITIES:
        raise ValueError(
            f"Unknown surf_type: {surf_type}. "
            f"Valid options: {list(VALID_DENSITIES.keys())}"
        )

    valid_densities = VALID_DENSITIES[surf_type]
    if density not in valid_densities:
        raise AssertionError(
            f"Invalid density '{density}' for {surf_type}. "
            f"Valid options: {valid_densities}"
        )

    if surf_type == 'fsaverage':
        return fetch_fsaverage(density=density)
    elif surf_type == 'fslr':
        return fetch_fslr(density=density)
    elif surf_type == 'civet':
        return fetch_civet(density=density)


def transform_volume_to_surface(ds, surf_type, target_density, method='linear'):
    """
    Transform volume data to surface representation.

    Parameters
    ----------
    ds : nibabel.Nifti1Image
        Volume data to transform
    surf_type : str
        Type of surface ('fsaverage', 'fslr', or 'civet')
    target_density : str
        Target mesh density
    method : str, default='linear'
        Interpolation method

    Returns
    -------
    tuple
        (data_lh, data_rh) - Left and right hemisphere surface data

    Raises
    ------
    ValueError
        If surf_type is not recognized
    """
    if surf_type not in _TRANSFORM_FUNCS:
        raise ValueError(
            f"Unknown surf_type: {surf_type}. "
            f"Valid options: {list(_TRANSFORM_FUNCS.keys())}"
        )

    transform_func, density_param = _TRANSFORM_FUNCS[surf_type]
    kwargs = {density_param: target_density, 'method': method}
    return transform_func(ds, **kwargs)


def setup_group_legend(ax, n_labels=2):
    """
    Setup legend for grouped plots, keeping only first n_labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify
    n_labels : int, default=2
        Number of legend labels to keep
    """
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= n_labels:
        ax.legend(handles[:n_labels], labels[:n_labels])


def set_collection_colors(ax, box_colors):
    """
    Set colors for stripplot collections.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the collections
    box_colors : list
        List of colors to apply
    """
    dot_colors = sum([[c] * 2 for c in box_colors], [])
    for collection, color in zip(ax.collections, dot_colors):
        collection.set_facecolor(color)


def chunkwise(iterable, size=2):
    """
    Split an iterable into chunks of specified size.

    Parameters
    ----------
    iterable : iterable
        Input iterable to chunk
    size : int, default=2
        Size of each chunk

    Returns
    -------
    zip
        Iterator of tuples with size elements each
    """
    it = iter(iterable)
    return zip(*[it] * size)


def connect_grouped_points(ax, alpha=0.15, linewidth=0.5, zorder=None):
    """
    Connect paired points between groups with lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the point collections
    alpha : float, default=0.15
        Transparency of connecting lines
    linewidth : float, default=0.5
        Width of connecting lines
    zorder : int, optional
        Z-order for the lines
    """
    for ax1, ax2 in chunkwise(ax.collections, size=2):
        for (x0, y0), (x1, y1) in zip(ax1.get_offsets(), ax2.get_offsets()):
            plot_kwargs = {
                'color': 'black',
                'alpha': alpha,
                'linewidth': linewidth
            }
            if zorder is not None:
                plot_kwargs['zorder'] = zorder
            ax.plot([x0, x1], [y0, y1], **plot_kwargs)
