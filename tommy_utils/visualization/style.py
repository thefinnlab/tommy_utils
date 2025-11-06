"""
Figure styling and appearance configuration.

This module provides functions for setting up consistent matplotlib and seaborn styling
across all visualizations.
"""

import matplotlib as mpl
import seaborn as sns


def figure_style(
    font_size=7,
    scatter_size=10,
    axes_color='black',
    font='Liberation Sans',
    fig_size=(4, 4),
    **kwargs
):
    """
    Set consistent Seaborn + Matplotlib figure style.

    Parameters
    ----------
    font_size : int
        Base font size for all text.
    scatter_size : int or float
        Default marker size for scatter plots.
    axes_color : str
        Color of all axes spines and ticks (default: black).
    font : str
        Font family to use (default: Helvetica).
    fig_size : tuple
        Default figure size (width, height) in inches.
    **kwargs :
        Additional rcParams for fine-tuning.
    """

    rc_defaults = {
        "font.size": font_size,
        "figure.figsize": fig_size,
        "figure.titlesize": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.5,
        "axes.edgecolor": axes_color,
        "axes.labelcolor": axes_color,
        "xtick.color": axes_color,
        "ytick.color": axes_color,
        "lines.linewidth": 1,
        "lines.markersize": scatter_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "savefig.transparent": True,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,
        "legend.frameon": False,
    }

    # Allow user to override rc defaults
    rc_defaults.update(kwargs)

    # Apply consistent Seaborn + Matplotlib theme
    sns.set_theme(style="ticks", context="paper", font=font, rc=rc_defaults)

    # Ensure vector font embedding compatibility
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['backend'] = 'QtAgg'
