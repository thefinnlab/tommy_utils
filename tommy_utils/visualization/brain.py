"""
Brain visualization utilities.

This module provides functions for visualizing neuroimaging data on brain surfaces
and volumes using nilearn, surfplot, and neuromaps.
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import nibabel as nib
from nilearn import plotting, image, glm
import numpy as np

from surfplot import Plot
from neuromaps.transforms import mni152_to_fslr, mni152_to_fsaverage, mni152_to_civet, _estimate_density, fsaverage_to_fsaverage
from neuromaps.datasets import fetch_fslr, fetch_fsaverage, fetch_civet
from collections import defaultdict


def plot_brain_volume(ds, vmax, title, cmap, out_fn=None):
    """
    Plot brain volume data using nilearn.

    Parameters
    ----------
    ds : nibabel.Nifti1Image
        Brain volume data to plot
    vmax : float
        Maximum value for color scale
    title : str
        Title for the plot
    cmap : str or colormap
        Colormap to use
    out_fn : str, optional
        Output filename for saving the plot
    """
    # zcoodinates for plotting
    coords = [range(-50,0,5), range(0,50,5)]

    # determine the display threshold based on the minimum of the masked betas
    vmin = np.nanmin(abs(ds.get_fdata()))

    # if the threshold is nan set it to the vmax (no values will be shown anyways)
    if np.isnan(vmin):
        vmin = vmax

    # create the colorbar for the image
    norm = TwoSlopeNorm(vmin=-vmax, vmax=vmax)
    threshold_cmap = threshold_cbar(cmap, norm, vmin)

    fig, axes = plt.subplots(2,1)

    for ax, coords in zip(axes, coords):
        ax = plotting.plot_stat_map(ds, threshold=vmin, cut_coords=coords, display_mode='z', draw_cross=False, axes=ax, colorbar=False, cmap=threshold_cmap)

    plt.suptitle(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], visible=False)
    fig = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=threshold_cmap), ax=cbar_ax)

    #save and clean up all opened figures
    if out_fn:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close('all')


def mask_nifti(img, mask):
    """
    Threshold a beta-value image by a thresholded stat mask.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to mask
    mask : nibabel.Nifti1Image or numpy.ndarray
        Mask to apply

    Returns
    -------
    nibabel.Nifti1Image
        Masked image
    """
    if nib.nifti1.Nifti1Image == type(mask):
        mask = mask.get_fdata()
    # Now grab mean image to threshold
    masked_img = img.get_fdata()

    masked_img[mask == 0] = np.nan

    return image.new_img_like(img, masked_img)


def threshold_tstats(tstats, alpha = .05, height_control = 'fdr', cluster_threshold = 0):
    """
    Apply statistical thresholding to t-statistics.

    Parameters
    ----------
    tstats : nibabel.Nifti1Image
        T-statistics image
    alpha : float, default=0.05
        Significance threshold
    height_control : str, default='fdr'
        Method for multiple comparison correction
    cluster_threshold : int, default=0
        Minimum cluster size

    Returns
    -------
    nibabel.Nifti1Image
        Thresholded image
    """
    # AFNI 2nd-level analyses include t-values and means, and we want to isolate the t-values for correction
    # Apply multiple comparisons corrections (FDR, q < .05) with cluster threshold for clean visualization
    thresholded_map, threshold = glm.threshold_stats_img(tstats,
                                                         alpha=alpha,
                                                         height_control=height_control,
                                                         cluster_threshold=cluster_threshold)
    print(f'The {height_control} = {alpha} threshold is %.3g' % threshold)

    return thresholded_map


def threshold_cbar(cmap, norm, threshold):
    """
    Create a thresholded colorbar.

    Parameters
    ----------
    cmap : str or colormap
        Colormap to threshold
    norm : matplotlib.colors.Normalize
        Normalization for the colormap
    threshold : float
        Threshold value

    Returns
    -------
    matplotlib.colors.ListedColormap
        Thresholded colormap
    """
    cmap = plt.get_cmap('RdBu_r')
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # set colors to grey for absolute values < threshold
    istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
    istop = int(norm(threshold, clip=True) * (cmap.N - 1))

    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)

    thresholded_cmap = ListedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    return thresholded_cmap


def plot_brain_values(ds, title, vmax, surf_mesh='fsaverage5', views=['lateral','medial'], out_fn=None, cmap='RdBu_r', colorbar=True):
    """
    Plot brain values on surface using nilearn.

    Parameters
    ----------
    ds : nibabel.Nifti1Image
        Brain data to plot
    title : str
        Title for the plot
    vmax : float
        Maximum value for color scale
    surf_mesh : str, default='fsaverage5'
        Surface mesh to use
    views : list, default=['lateral','medial']
        Views to display
    out_fn : str, optional
        Output filename for saving the plot
    cmap : str, default='RdBu_r'
        Colormap to use
    colorbar : bool, default=True
        Whether to show colorbar
    """
    fig = plt.figure()
    fig = plotting.plot_img_on_surf(ds,
                                surf_mesh=surf_mesh,
                                views=views,
                                inflate=True,
                                threshold=0,
                                vmax=vmax,
                                title=title,
                                cmap=cmap,
                                symmetric_cbar=True,
                                colorbar=colorbar)

    #save and clean up all opened figures
    if out_fn:
        plt.savefig(out_fn, bbox_inches='tight', transparent=True, dpi=300)
        plt.close('all')


def make_layers_dict(data, cmap, alpha=0.75, label=None, color_range=None, cbar=True):
    """
    Create a dictionary for surfplot layer configuration.

    Parameters
    ----------
    data : dict
        Dictionary with 'left' and 'right' hemisphere data
    cmap : str or colormap
        Colormap to use
    alpha : float, default=0.75
        Transparency level
    label : str, optional
        Label for the colorbar
    color_range : tuple, optional
        (min, max) for color scale
    cbar : bool, default=True
        Whether to show colorbar for this layer

    Returns
    -------
    dict
        Layer configuration dictionary
    """
    d = defaultdict()
    d['data'] = data
    d['cmap'] = cmap
    d['alpha'] = alpha
    d['label'] = label
    d['color_range'] = color_range
    d['cbar'] = cbar

    return d


def sigmoid(x):
    """
    Sigmoid activation function.

    Parameters
    ----------
    x : numpy.ndarray
        Input array

    Returns
    -------
    numpy.ndarray
        Sigmoid-transformed array
    """
    return 1 / (1 + np.exp(-x))


def create_depth_map(surf_type='fsaverage', target_density='41k'):
    """
    Create a depth map for brain surface visualization.

    Parameters
    ----------
    surf_type : str, default='fsaverage'
        Type of surface ('fsaverage', 'fslr', or 'civet')
    target_density : str, default='41k'
        Target mesh density

    Returns
    -------
    dict
        Layer configuration dictionary for depth map
    """
    if surf_type == 'fsaverage':
        assert (target_density in ['3k', '10k', '41k', '164k'])
        surfaces = fetch_fsaverage(density=target_density)
    elif surf_type == 'fslr':
        assert (target_density in ['4k', '8k', '32k', '164k'])
        surfaces = fetch_fslr(density=target_density)
    elif surf_type == 'civet':
        assert (target_density in ['41k', '164k'])
        surfaces = fetch_civet(density=target_density)

    # create the cmap for the depth map
    cmap = plt.get_cmap('Greys_r')
    cmap = cmap(np.arange(0,256))
    cmap = ListedColormap(cmap)

    left = sigmoid(nib.load(surfaces['sulc'][0]).agg_data())
    right = sigmoid(nib.load(surfaces['sulc'][1]).agg_data())

    depth = make_layers_dict(data={'left': left, 'right': right},
        cmap=cmap, alpha=1, color_range=(0, 1), cbar=False)

    return depth


def vol_to_surf(ds, surf_type='fsaverage', map_type='inflated', target_density='41k', method='linear'):
    """
    Convert volume data to surface representation.

    Parameters
    ----------
    ds : nibabel.Nifti1Image
        Volume data to convert
    surf_type : str, default='fsaverage'
        Type of surface ('fsaverage', 'fslr', or 'civet')
    map_type : str, default='inflated'
        Type of surface map
    target_density : str, default='41k'
        Target mesh density
    method : str, default='linear'
        Interpolation method

    Returns
    -------
    tuple
        (surfaces, data) where surfaces are mesh files and data is dict with 'left'/'right' hemispheres
    """
    if surf_type == 'fsaverage':
        assert (target_density in ['3k', '10k', '41k', '164k'])
        surfaces = fetch_fsaverage(density=target_density)
        data_lh, data_rh = mni152_to_fsaverage(ds, fsavg_density=target_density, method=method)
    elif surf_type == 'fslr':
        assert (target_density in ['4k', '8k', '32k', '164k'])
        surfaces = fetch_fslr(density=target_density)
        data_lh, data_rh = mni152_to_fslr(ds, fslr_density=target_density, method=method)
    elif surf_type == 'civet':
        assert (target_density in ['41k', '164k'])
        surfaces = fetch_civet(density=target_density)
        data_lh, data_rh = mni152_to_civet(ds, civet_density=target_density, method=method)

    surfs = surfaces[map_type]
    data = {'left': data_lh, 'right': data_rh}

    return surfs, data


def numpy_to_surface(ds, surf_type='fsaverage', map_type='inflated', target_density='41k', method='linear'):
    """
    Takes a numpy array surface and makes a gifti surface ready for plotting.

    Parameters
    ----------
    ds : numpy.ndarray
        Surface data as numpy array
    surf_type : str, default='fsaverage'
        Type of surface ('fsaverage', 'fslr', or 'civet')
    map_type : str, default='inflated'
        Type of surface map
    target_density : str, default='41k'
        Target mesh density
    method : str, default='linear'
        Interpolation method for resampling

    Returns
    -------
    tuple
        (surfaces, data) where surfaces are mesh files and data is dict with 'left'/'right' hemispheres
    """
    ds = ds.astype('float32')
    hemis = np.split(ds, 2)

    data = []

    for hemi in hemis:
        # Create new surface data objects for left and right hemispheres
        surf = nib.gifti.GiftiImage()
        surf_array = nib.gifti.GiftiDataArray(hemi.squeeze(), intent='NIFTI_INTENT_SHAPE', datatype='NIFTI_TYPE_FLOAT32')
        surf.add_gifti_data_array(surf_array)
        data.append(surf)

    data = tuple(data)
    density, = _estimate_density((data,), hemi=None)

    if density != target_density:
        data = fsaverage_to_fsaverage(data, target_density=target_density, method=method)
        density = target_density

    if surf_type == 'fsaverage':
        assert (density in ['3k', '10k', '41k', '164k'])
        surfaces = fetch_fsaverage(density)
    elif surf_type == 'fslr':
        assert (density in ['4k', '8k', '32k', '164k'])
        surfaces = fetch_fslr(density=density)
    elif surf_type == 'civet':
        assert (density in ['41k', '164k'])
        surfaces = fetch_civet(density=density)

    surfs = surfaces[map_type]
    data_lh, data_rh = data
    data = {'left': data_lh, 'right': data_rh}

    return surfs, data


def plot_surf_data(surfs, layers_info, surf_type='fslr', views=['lateral', 'medial'], zoom=1.35, brightness=0.8, scale=(10,10),
    surf_alpha=1, add_depth=False, embed_nb=False, colorbar=True, cbar_loc=None, title=None, out_fn=None):
    """
    Plot surface data using surfplot.

    Parameters
    ----------
    surfs : tuple
        Surface mesh files (left, right)
    layers_info : list
        List of layer configuration dictionaries
    surf_type : str, default='fslr'
        Type of surface ('fsaverage', 'fslr', or 'civet')
    views : list, default=['lateral', 'medial']
        Views to display
    zoom : float, default=1.35
        Zoom level
    brightness : float, default=0.8
        Brightness level
    scale : tuple, default=(10,10)
        Figure scale
    surf_alpha : float, default=1
        Surface transparency
    add_depth : bool, default=False
        Whether to add depth map layer
    embed_nb : bool, default=False
        Whether to embed in notebook
    colorbar : bool, default=True
        Whether to show colorbar
    cbar_loc : str, optional
        Colorbar location
    title : str, optional
        Title for the plot
    out_fn : str, optional
        Output filename for saving the plot

    Returns
    -------
    tuple
        (fig, plot) matplotlib figure and surfplot Plot object
    """
    if len(views) == 1:
        zoom=2.35
        scale=(10, 5)

    p = Plot(*surfs, views=views, zoom=zoom, embed_nb=embed_nb, brightness=brightness, surf_alpha=surf_alpha)

    # if we want to add depth insert into the start of the list
    if add_depth:
        density_est = (layers_info[0]['data']['left'], layers_info[0]['data']['right'])
        density, = _estimate_density((density_est,), hemi=None)
        depth = create_depth_map(surf_type=surf_type, target_density=density)
        layers_info.insert(0, depth)

    for layer in layers_info:
        p.add_layer(data=layer['data'],
                    cmap=layer['cmap'],
                    cbar_label=layer['label'],
                    alpha=layer['alpha'],
                    color_range=layer['color_range'],
                    cbar=layer['cbar']
                     )

    if cbar_loc == 'right':
        kws = {'location': 'right', 'label_direction': 45, 'decimals': 1,
                 'fontsize': 8, 'n_ticks': 2, 'shrink': .15, 'aspect': 8,
                 'draw_border': False}
    else:
        kws = {'aspect': 10}

    fig = p.build(cbar_kws=kws, scale=scale, colorbar=colorbar)

    if title:
        plt.title(title)

    #save and clean up all opened figures
    if out_fn:
        fig.savefig(out_fn, bbox_inches='tight', transparent=True, dpi=300)
        plt.close('all')

    return fig, p


def combine_images(columns, images, row_names=None, column_names=None, title=None, legend=None, fig_size=(16, 4), pad=5, out_fn=None):
    """
    Combine multiple images into a grid layout.

    Parameters
    ----------
    columns : int
        Number of columns in the grid
    images : list
        List of image file paths
    row_names : list, optional
        Labels for rows
    column_names : list, optional
        Labels for columns
    title : str, optional
        Overall title for the figure
    legend : str, optional
        Legend text to display
    fig_size : tuple, default=(16, 4)
        Figure size (width, height)
    pad : int, default=5
        Padding between subplots
    out_fn : str, optional
        Output filename for saving the plot

    Returns
    -------
    matplotlib.figure.Figure
        The combined figure
    """
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    fig = plt.figure(1, figsize=fig_size, constrained_layout=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, columns),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
        im = plt.imread(im)
        ax.imshow(im)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.setp(ax.spines.values(), visible=False)

    if row_names:
        for ax, row in zip(np.array(grid.axes_row)[:,0], row_names): # np.array(grid.axes_row)[:,0]
#             ax.set_title(row)
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                                xycoords=ax.yaxis.label, textcoords='offset points',
                                ha='right', va='center', fontsize=fig_size[0])
    if column_names:
        for ax, col in zip(np.array(grid.axes_row)[0], column_names): # np.array(grid.axes_row)[0]
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                                xycoords='axes fraction', textcoords='offset points',
                                ha='center', va='baseline', fontsize=fig_size[0])

    # fig.tight_layout()

    if legend:
        fig.text(x=0.95, y=0.4,
                         s=legend,
                         fontsize=fig_size[0],
                         bbox=dict(facecolor='none', edgecolor='black', pad=5.0))

    if title:
        if row_names == None:
            row_names = [None]

            #0.1375
        fig.suptitle(title, x=0.5, y=0.5 + 0.1125 * len(row_names), fontsize=fig_size[0]+6)

    if out_fn:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close('all')

    return fig


def draw_umap(data, colors, n_neighbors=15, min_dist=0.1, random_state=42, n_components=2, metric='euclidean', title='', s=10, cmap='jet'):
    """
    Create a UMAP visualization.

    Note: This function requires the umap-learn package to be installed.

    Parameters
    ----------
    data : numpy.ndarray
        Input data for UMAP
    colors : array-like
        Colors for each data point
    n_neighbors : int, default=15
        UMAP n_neighbors parameter
    min_dist : float, default=0.1
        UMAP min_dist parameter
    random_state : int, default=42
        Random seed for reproducibility
    n_components : int, default=2
        Number of UMAP components (1, 2, or 3)
    metric : str, default='euclidean'
        Distance metric to use
    title : str, default=''
        Plot title
    s : int, default=10
        Marker size
    cmap : str, default='jet'
        Colormap for points

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    import umap

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state
    )
    u = fit.fit_transform(data);

    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax = ax.scatter(u[:,0], range(len(u)), c=colors, s=s, cmap=cmap)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax = ax.scatter(u[:,0], u[:,1], c=colors, s=s, cmap=cmap) #, edgecolors=(0.5,0.5,0.5))
    if n_components == 3:
#         ax = plt.Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax = ax.scatter(u[:,0], u[:,1], u[:,2], c=colors, s=s, cmap=cmap)
    plt.title(title, fontsize=18)

    return fig, ax


def plot_colorbar(vmin, vmax, nticks=5, direction='horizontal', cmap='RdBu_r', out_fn=None):
    """
    Plot a standalone colorbar.

    Parameters
    ----------
    vmin : float
        Minimum value for colorbar
    vmax : float
        Maximum value for colorbar
    nticks : int, default=5
        Number of ticks on colorbar
    direction : str, default='horizontal'
        Orientation ('horizontal' or 'vertical')
    cmap : str, default='RdBu_r'
        Colormap to use
    out_fn : str, optional
        Output filename for saving the plot
    """
    fig = plt.figure()

    values = np.random.randn(10)

    divnorm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    psm = plt.pcolormesh([-values, values], norm=divnorm, cmap=cmap)
    plt.clf()

    # xloc, yloc, size x, size y
    if direction == 'horizontal':
        cbar_ax = fig.add_axes([0.5, 0, 0.6, 0.05])
    elif direction == 'vertical':
        cbar_ax = fig.add_axes([0.5, 0, 0.05, 0.6])

    fig.colorbar(psm, cax=cbar_ax, orientation=direction, ticks=ticker.MaxNLocator(nbins=nticks))

    if out_fn:
        plt.savefig(out_fn, bbox_inches='tight', transparent=True)
