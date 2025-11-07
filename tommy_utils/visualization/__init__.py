"""
Visualization subpackage for tommy_utils.

This subpackage provides utilities for creating publication-quality visualizations
including statistical plots, brain surface/volume plots, and figure styling.

Modules
-------
style : Figure styling utilities
plots : Statistical and general plotting functions
brain : Brain visualization functions
"""

# Import from style module
from .style import figure_style

# Import from plots module
from .plots import (
    chunkwise,
    scatter_boxplot,
    kde_boxplot,
    scatter_barplot,
    plot_correlation_matrices,
)

# Import from brain module
from .brain import (
    plot_brain_volume,
    plot_brain_values,
    vol_to_surf,
    numpy_to_surface,
    plot_surf_data,
    make_layers_dict,
    create_depth_map,
    mask_nifti,
    threshold_tstats,
    threshold_cbar,
    sigmoid,
    combine_images,
    draw_umap,
    plot_colorbar,
)

__all__ = [
    # Style
    'figure_style',
    # Plots
    'chunkwise',
    'scatter_boxplot',
    'kde_boxplot',
    'scatter_barplot',
    'plot_correlation_matrices',
    # Brain
    'plot_brain_volume',
    'plot_brain_values',
    'vol_to_surf',
    'numpy_to_surface',
    'plot_surf_data',
    'make_layers_dict',
    'create_depth_map',
    'mask_nifti',
    'threshold_tstats',
    'threshold_cbar',
    'sigmoid',
    'combine_images',
    'draw_umap',
    'plot_colorbar',
]
