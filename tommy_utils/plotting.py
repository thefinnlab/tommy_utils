import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, ListedColormap

from matplotlib.patches import PathPatch

from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import nibabel as nib
from nilearn import plotting, image, glm
import numpy as np
import sys

from surfplot import Plot
from neuromaps.transforms import mni152_to_fslr, mni152_to_fsaverage, mni152_to_civet, _estimate_density, fsaverage_to_fsaverage
from neuromaps.datasets import fetch_fslr, fetch_fsaverage, fetch_civet
from collections import defaultdict
# import umap

############################################
############# FIGURE SETUP #################
############################################

# Set consistent Seaborn + Matplotlib figure style.
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

    # colors = {
    #     'hit': sns.color_palette('Set2')[0],
    #     'miss': sns.color_palette('Set2')[1],
    # }

    # return colors

def combine_images(columns, images, row_names=None, column_names=None, title=None, legend=None, fig_size=(16, 4), pad=5, out_fn=None):
		
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

############################################
############ UMAP PLOTTING ################
############################################

def draw_umap(data, colors, n_neighbors=15, min_dist=0.1, random_state=42, n_components=2, metric='euclidean', title='', s=10, cmap='jet'):
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

############################################
############ PAIRED PLOTTING ###############
############################################

# for "pairs" of any length
def chunkwise(t, size=2):
	it = iter(t)
	return zip(*[it]*size)

def scatter_boxplot(df, x, y, group=None, palette='RdBu_r', order=None, ax=None, use_legend=False):
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
	"""Create a bar plot with individual data points.
	
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
	
	fig = plt.figure()
	
	for i in range(n_values):
		plt.subplot(1, n_values, i+1)
		
		sns.heatmap(matrices[idxs][i], square=True, cbar_kws={"shrink": .25}, cmap='RdBu_r', vmax=vmax, vmin=-vmax)

		plt.title(f'{labels[idxs[i]]}')
		
	plt.suptitle(title, y=0.75)
	plt.tight_layout()

	plt.savefig(out_fn, bbox_inches='tight')
	plt.close('all')

############################################
############ NILEARN PLOTTING ##############
############################################

def plot_brain_volume(ds, vmax, title, cmap, out_fn=None):
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
	'''
	Threshold a beta-value image by a thresholded stat mask.
	'''
	
	if nib.nifti1.Nifti1Image == type(mask):
		mask = mask.get_fdata()
	# Now grab mean image to threshold
	masked_img = img.get_fdata()
	
	masked_img[mask == 0] = np.nan
	
	return image.new_img_like(img, masked_img)

def threshold_tstats(tstats, alpha = .05, height_control = 'fdr', cluster_threshold = 0):
	
	# AFNI 2nd-level analyses include t-values and means, and we want to isolate the t-values for correction
	# Apply multiple comparisons corrections (FDR, q < .05) with cluster threshold for clean visualization
	thresholded_map, threshold = glm.threshold_stats_img(tstats, 
														 alpha=alpha, 
														 height_control=height_control,
														 cluster_threshold=cluster_threshold)
	print(f'The {height_control} = {alpha} threshold is %.3g' % threshold)
	
	return thresholded_map

def threshold_cbar(cmap, norm, threshold):
	'''
	Create a thresholded colorbar
	'''
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


############################################
############ SURFPLOT PLOTTING #############
############################################

def make_layers_dict(data, cmap, alpha=0.75, label=None, color_range=None, cbar=True):

	d = defaultdict()
	d['data'] = data
	d['cmap'] = cmap
	d['alpha'] = alpha
	d['label'] = label
	d['color_range'] = color_range
	d['cbar'] = cbar
	
	return d

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def create_depth_map(surf_type='fsaverage', target_density='41k'):

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
	'''
	Takes a numpy array surface and makes a gifti surface ready
	for plotting 
	'''

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