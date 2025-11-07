"""
Utilities for working with brain atlases.

This module provides functions for loading, combining, and manipulating brain atlases.
It supports various atlas formats and handles region overlaps and priorities.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img

# Import visualization for surface transformations
from ..visualization import vol_to_surf, numpy_to_surface

# Get the data directory path relative to this module
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_MODULE_DIR, 'data', 'atlases')

# Constants
ATLAS_REGIONS = {
    'glasser': {
        'sensory_auditory': ['Early auditory', 'A4', 'A5'],
        'early_visual': ['V1', 'V2', 'V3'],
        'late_visual': ['LOC', 'FFC', 'PHA'],
        'language': ['PSL', 'STG', 'STS', 'TE1', 'IFJ', 'IFS'],
        'multimodal': ['PostCin', 'AntCin', 'POS', 'TPOJ', 'mPFC']
    }
}

def load_combined_atlas(target_density='41k', prob_threshold=0.3, overlap_threshold=0.5):
    """Load and combine multiple brain atlases into a single unified atlas.

    This function loads multiple atlases (Fedorenko, Glasser, and visual ROIs) and combines them
    into a single atlas based on priority order. The combined atlas preserves the original area
    names while handling overlapping regions according to the specified priority.

    Parameters:
    -----------
    target_density : str, optional
        Target surface density (default: '41k')
    prob_threshold : float, optional
        Threshold for probability maps (default: 0.3)
    overlap_threshold : float, optional
        Threshold for considering regions as overlapping (default: 0.5)

    Returns:
    --------
    tuple
        (combined_atlas, combined_labels)
        - combined_atlas: numpy array containing the combined atlas data
        - combined_labels: pandas.DataFrame with columns:
            - Parcel Index: unique label in combined atlas
            - Aggregate Label: atlas name (e.g., 'ToM', 'language', etc.)
            - Area Names: comma-separated list of original area names
    """

    # Step 1: Load Fedorenko atlases (language, ToM, MD)
    # These are probability-based atlases that are thresholded to create binary masks
    fedorenko_atlases = ['language', 'ToM', 'MD']
    all_atlases = [
        load_fedorenko_atlas(
            atlas_type=atlas,
            target_density=target_density,
            return_type='thresholded',
            prob_threshold=prob_threshold  # Threshold for probability maps
        ) for atlas in fedorenko_atlases
    ]
    all_atlases, labels = [list(x) for x in zip(*all_atlases)]

    # Step 2: Load Glasser atlas
    # This is a structural atlas with predefined regions
    glasser, glasser_labels = load_glasser_atlas(
        atlas_format='surface',
        target_density=target_density,
        data_dir=DATA_DIR,
        regions=['Early auditory', 'A4', 'A5']
    )

    # Step 3: Load visual ROIs atlas
    # This contains visual stream regions from NSD dataset
    visual_atlas, visual_labels = load_visual_rois(
        atlas_type='nsd_streams',
        target_density=target_density,
        data_dir=DATA_DIR
    )

    # Step 4: Prepare atlases for combination
    # Order matters for the combination process
    atlas_names = ['visual'] + ['glasser'] + fedorenko_atlases
    all_atlases = [visual_atlas] + [glasser] + all_atlases
    labels = [visual_labels] + [glasser_labels] + labels

    atlas_priority = ['ToM', 'MD', 'language', 'visual', 'glasser']
    # Step 5: Combine atlases with priority order
    # Priority determines which atlas takes precedence in overlapping regions
    # Higher priority atlases (earlier in list) override lower priority ones
    combined_atlas, combined_labels = combine_atlases(
        all_atlases,
        atlas_names=atlas_names,
        atlas_labels=labels,
        overlap_threshold=overlap_threshold,  # Full overlap required for replacement
        atlas_priority=atlas_priority # Priority order
    )

    return combined_atlas, combined_labels

# Helper Functions
def _get_region_label(label_df, label, atlas_name):
    """Get the region label from the label DataFrame.

    Args:
        label_df (pd.DataFrame): DataFrame containing label mappings
        label (int): Region label to look up
        atlas_name (str): Name of the source atlas

    Returns:
        str: Region label
    """
    if label_df is None:
        return atlas_name

    label_info = label_df[label_df['Parcel Index'] == int(label)]
    if not label_info.empty:
        return label_info['Aggregate Label'].iloc[0]
    return f"unknown_{label}"

def _find_overlapping_regions(atlas_list, atlas_names, atlas_labels, overlap_threshold=0.5):
    """Find all regions and their overlaps across atlases.

    Args:
        atlas_list (list): List of numpy arrays, each representing an atlas
        atlas_names (list): List of names for each atlas
        atlas_labels (list): List of label DataFrames for each atlas
        overlap_threshold (float): Threshold for considering regions as overlapping

    Returns:
        tuple: (regions, overlaps)
            - regions: dict mapping region labels to (atlas_name, label) tuples
            - overlaps: dict mapping region labels to list of overlapping region labels
    """
    # Initialize dictionaries to store regions and their overlaps
    regions = {}  # Maps region_label -> (atlas_name, label)
    overlaps = {}  # Maps region_label -> [overlapping_region_labels]

    # Step 1: Collect all regions from each atlas
    for atlas_idx, (atlas, atlas_name) in enumerate(zip(atlas_list, atlas_names)):
        label_df = atlas_labels[atlas_idx] if atlas_labels is not None else None

        # Get unique non-zero labels in this atlas
        unique_labels = np.unique(atlas)
        unique_labels = unique_labels[unique_labels != 0]

        # Store each region
        for label in unique_labels:
            region_label = _get_region_label(label_df, label, atlas_name)
            regions[region_label] = (atlas_name, label)
            overlaps[region_label] = []

    # Step 2: Find overlaps between regions
    for atlas_idx, (atlas, atlas_name) in enumerate(zip(atlas_list, atlas_names)):
        label_df = atlas_labels[atlas_idx] if atlas_labels is not None else None

        # Get unique non-zero labels in this atlas
        unique_labels = np.unique(atlas)
        unique_labels = unique_labels[unique_labels != 0]

        # Check each region for overlaps with other atlases
        for label in unique_labels:
            region_label = _get_region_label(label_df, label, atlas_name)
            region_mask = (atlas == label)

            # Check overlap with all other atlases
            for other_idx, (other_atlas, other_name) in enumerate(zip(atlas_list, atlas_names)):
                if other_idx == atlas_idx:
                    continue

                other_label_df = atlas_labels[other_idx] if atlas_labels is not None else None

                # Check overlap with each region in other atlas
                for other_label in np.unique(other_atlas)[1:]:  # Skip 0
                    other_mask = (other_atlas == other_label)
                    other_region_label = _get_region_label(other_label_df, other_label, other_name)

                    # Calculate overlap ratio in both directions
                    overlap_ratio_forward = np.sum(region_mask & other_mask) / np.sum(region_mask)
                    overlap_ratio_backward = np.sum(region_mask & other_mask) / np.sum(other_mask)

                    # Record overlap if either direction exceeds threshold
                    if overlap_ratio_forward >= overlap_threshold or overlap_ratio_backward >= overlap_threshold:
                        # Record overlap in both directions
                        if other_region_label not in overlaps[region_label]:
                            overlaps[region_label].append(other_region_label)
                        if region_label not in overlaps[other_region_label]:
                            overlaps[other_region_label].append(region_label)

    return regions, overlaps

# Atlas Loading Functions
def load_fedorenko_probabilities(atlas_type='language', target_density='10k', data_dir=None):
    """Load raw probability maps from Fedorenko atlas.

    Args:
        atlas_type (str): Type of atlas to load ('language', 'MD', or 'ToM')
        target_density (str): Target density for surface data (e.g. '10k', '32k', etc)
        data_dir (str): Base directory containing atlas files (default: module data directory)

    Returns:
        tuple: (probabilities, labels)
            - probabilities: numpy array with probability values
            - labels: pandas.DataFrame with columns 'Parcel Index' and 'Aggregate Label'
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Dictionary mapping atlas types to their filenames
    atlas_fns = {
        'language': {
            'prob_top10': {'lh': 'LH_LanA_n804.nii.gz',
                          'rh': 'RH_LanA_n804.nii.gz'}
        },
        'MD': {
            'prob_top10': 'MDloc_n691_top10%_atlas.nii'
        },
        'ToM': {
            'prob_top10': 'ToMloc_n198_top10%_atlas.nii'
        }
    }

    data_dir = os.path.join(data_dir, 'fedorenko', atlas_type)

    if atlas_type == 'language':
        # Load left and right hemisphere language atlas files
        hemis = []
        for hemi, fn in atlas_fns[atlas_type]['prob_top10'].items():
            fn = os.path.join(data_dir, fn)
            hemis.append(nib.load(fn).get_fdata().squeeze())

        # Load and concatenate data
        probabilities = np.concatenate(hemis)

    elif atlas_type in ['MD', 'ToM']:
        prob_top10_fn = os.path.join(data_dir, atlas_fns[atlas_type]['prob_top10'])
        atlas_data = nib.load(prob_top10_fn)

        # Convert volume to surface
        surfs, data = vol_to_surf(atlas_data, target_density=target_density)
        probabilities = np.concatenate([v.darrays[0].data for k, v in data.items()])

    else:
        raise ValueError(f"Unknown atlas type: {atlas_type}")

    if target_density is not None:
        surfs, data = numpy_to_surface(probabilities, target_density=target_density)
        probabilities = np.concatenate([v.darrays[0].data for k, v in data.items()])

    # Create labels DataFrame
    labels = pd.DataFrame({
        'Parcel Index': [1],
        'Aggregate Label': [f'{atlas_type}_network_probabilities']
    })

    return probabilities, labels

def load_fedorenko_parcels(atlas_type='language', target_density='10k', data_dir=None):
    """Load parcel labels from Fedorenko atlas.

    Args:
        atlas_type (str): Type of atlas to load ('language', 'MD', or 'ToM')
        target_density (str): Target density for surface data (e.g. '10k', '32k', etc)
        data_dir (str): Base directory containing atlas files (default: module data directory)

    Returns:
        tuple: (parcels, labels)
            - parcels: numpy array with parcel indices
            - labels: pandas.DataFrame with columns 'Parcel Index' and 'Aggregate Label'
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Dictionary mapping atlas types to their filenames
    atlas_fns = {
        'language': {
            'parcels': 'allParcels_language_SN220.nii',
            'parcel_labels': 'allParcels_language_SN220.txt'
        },
        'MD': {
            'parcels': 'allParcels-MD-HE197.nii',
            'parcel_labels': 'allParcels-MD-HE197.txt'
        },
        'ToM': {
            'parcels': 'allParcels_ToM.nii',
            'parcel_labels': 'allParcels_ToM.txt'
        }
    }

    data_dir = os.path.join(data_dir, 'fedorenko', atlas_type)

    # Load parcel labels
    labels_fn = os.path.join(data_dir, atlas_fns[atlas_type]['parcel_labels'])
    with open(labels_fn, 'r') as f:
        parcel_labels = [line.strip() for line in f.readlines() if line.strip()]

    # Load original parcels
    if atlas_type == 'language':
        parcels_fn = os.path.join(data_dir, atlas_fns[atlas_type]['parcels'])
        parcels_nii = nib.load(parcels_fn)
        _, data = vol_to_surf(parcels_nii, target_density=target_density, method='nearest')
        parcels = np.concatenate([v.darrays[0].data for k, v in data.items()])
    else:
        parcels_fn = os.path.join(data_dir, atlas_fns[atlas_type]['parcels'])
        parcels_nii = nib.load(parcels_fn)
        _, data = vol_to_surf(parcels_nii, target_density=target_density, method='nearest')
        parcels = np.concatenate([v.darrays[0].data for k, v in data.items()])

    # Create labels DataFrame
    labels = pd.DataFrame({
        'Parcel Index': range(1, len(parcel_labels) + 1),
        'Aggregate Label': parcel_labels
    })

    return np.round(parcels), labels

def load_fedorenko_atlas(atlas_type='language', target_density='10k', prob_threshold=0.25, return_type='thresholded', data_dir=None):
    """Load Fedorenko atlas data with options for different return types.

    Args:
        atlas_type (str): Type of atlas to load ('language', 'MD', or 'ToM')
        target_density (str): Target density for surface data (e.g. '10k', '32k', etc)
        prob_threshold (float): Threshold value below which data will be set to 0 (only used if return_type is 'thresholded')
        return_type (str): Type of data to return:
            - 'probabilities': Raw probability maps
            - 'parcels': Original parcel labels
            - 'thresholded': Parcels after thresholding by probabilities
        data_dir (str): Base directory containing atlas files (default: module data directory)

    Returns:
        tuple: (atlas_data, labels)
            - atlas_data: numpy array with requested data type
            - labels: pandas.DataFrame with columns 'Parcel Index' and 'Aggregate Label'
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if return_type == 'probabilities':
        return load_fedorenko_probabilities(atlas_type, target_density, data_dir)

    elif return_type == 'parcels':
        return load_fedorenko_parcels(atlas_type, target_density, data_dir)

    elif return_type == 'thresholded':
        # Load probabilities and threshold them
        probs, _ = load_fedorenko_probabilities(atlas_type, target_density, data_dir)
        probs[probs < prob_threshold] = 0

        # Load original parcel labels for reference
        parcels, parcel_labels = load_fedorenko_parcels(atlas_type, target_density, data_dir)

        # Threshold parcels by probabilities
        parcels[probs == 0] = 0

        return parcels, parcel_labels

    else:
        raise ValueError(f"Unknown return_type: {return_type}")

def load_visual_rois(atlas_type='nsd_streams', target_density='10k', regions=None, data_dir=None):
    """Load visual ROI atlases from MGZ files.

    Args:
        atlas_type (str): Type of visual ROIs to load ('nsd_streams' or 'Kastner2015')
        target_density (str): Target density for surface data (e.g. '10k', '32k', etc)
        regions (list, optional): List of region names to include. If None, includes all regions.
        data_dir (str): Directory containing atlas files (default: module data directory)

    Returns:
        tuple: (atlas, labels)
            - atlas: Combined left and right hemisphere atlas data
            - labels: pandas.DataFrame with columns 'Parcel Index' and 'Aggregate Label'
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Set atlas-specific paths and files
    if atlas_type == 'nsd_streams':
        label_file = 'streams.mgz.ctab'
        atlas_prefix = 'streams'
    elif atlas_type == 'Kastner2015':
        label_file = 'Kastner2015.mgz.ctab'
        atlas_prefix = 'Kastner2015'
    else:
        raise ValueError(f"Unknown atlas_type: {atlas_type}. Must be 'nsd_streams' or 'Kastner2015'")

    # Load label mappings
    mapping_df = pd.read_csv(os.path.join(data_dir, atlas_type, label_file),
                            delimiter=' ', header=None, index_col=0)

    # Convert to consistent DataFrame format
    labels = pd.DataFrame({
        'Parcel Index': mapping_df.index,
        'Aggregate Label': mapping_df[1]
    }).reset_index(drop=True)

    # Filter labels if specific regions requested
    if regions is not None:
        labels = labels[labels['Aggregate Label'].isin(regions)]
        if len(labels) == 0:
            raise ValueError(f"None of the requested regions {regions} were found")

    # Load data for each hemisphere
    atlas = {}
    for hemi in ['lh', 'rh']:
        mgz_file = os.path.join(data_dir, atlas_type, f'{hemi}.{atlas_prefix}.mgz')
        atlas[hemi] = nib.load(mgz_file).get_fdata()

    # Combine hemispheres
    atlas_data = np.squeeze(np.vstack([atlas['lh'], atlas['rh']]))

    # Filter atlas data to only include requested regions
    if regions is not None:
        mask = np.isin(atlas_data, labels['Parcel Index'])
        atlas_data[~mask] = 0

    if target_density is not None:
        surfs, data = numpy_to_surface(atlas_data, target_density=target_density, method='nearest')
        atlas_data = np.concatenate([v.darrays[0].data for k, v in data.items()])

    return atlas_data, labels

def load_glasser_atlas(atlas_format='surface', target_density='10k', nifti_fn=None, data_dir=None, regions=None, aggregate_column='Aggregate Label'):
    """Load Glasser atlas in either surface or volume format.

    Args:
        atlas_format (str): Format to load atlas in - either 'surface' or 'volume'
        target_density (str): Target density for surface data (e.g. '10k', '32k', etc)
        nifti_fn (str, optional): Path to reference nifti file for resampling volume atlas
        data_dir (str): Directory containing atlas files (default: module data directory)
        regions (list, optional): List of region names to include. If None, includes all regions.
        aggregate_column (str, optional): Column name to use for aggregating regions. Default is 'Aggregate Label'.

    Returns:
        tuple: (atlas, labels)
            - atlas: Atlas data in requested format
            - labels: pandas.DataFrame with columns:
                - Parcel Index: unique label in combined atlas
                - [aggregate_column]: aggregate label used for grouping
                - Area Names: comma-separated list of original area names
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Load and process labels
    labels = pd.read_csv(os.path.join(data_dir, 'glasser', 'Glasser_Aggregate_Mapping.csv'))

    # Clean up aggregate labels
    labels[aggregate_column] = labels[aggregate_column].str.split('(').str[0].str.strip()
    labels.loc[labels[aggregate_column].isin(['STGa', 'STGd']), aggregate_column] = 'STG'
    labels.loc[labels[aggregate_column].isin(['STSv']), aggregate_column] = 'STS'

    # Filter labels if specific regions requested
    if regions is not None:
        labels = labels[labels[aggregate_column].isin(regions)]
        if len(labels) == 0:
            raise ValueError(f"None of the requested regions {regions} were found")

    if atlas_format == 'surface':
        from neuromaps.images import annot_to_gifti
        from neuromaps.transforms import fsaverage_to_fsaverage

        # Process both hemispheres
        hemis = {'left': 'L', 'right': 'R'}
        glasser = {}

        for hemi, code in hemis.items():
            # Load annotation file
            annot_file = os.path.join(data_dir, 'glasser', f'{hemi[0]}h.HCP-MMP1.annot')

            # Convert to GIFTI
            gifti = annot_to_gifti(parcellation=(annot_file,))[0]
            glasser[f'map_{hemi}'] = gifti.darrays[0].data

            # Downsample to target density
            gifti_resampled = fsaverage_to_fsaverage(gifti, target_density=target_density,
                                              hemi=code, method='nearest')[0]

            glasser[f'map_{hemi}'] = gifti_resampled.darrays[0].data

        atlas_data = np.concatenate([glasser['map_left'], glasser['map_right']])

        # Filter atlas data to only include requested regions
        if regions is not None:
            mask = np.isin(atlas_data, labels['Parcel Index'])
            atlas_data[~mask] = 0

        # Create mapping from aggregate labels to parcel indices and area names
        label_to_indices = {}
        label_to_areas = {}
        for _, row in labels.iterrows():
            agg_label = row[aggregate_column]
            if agg_label not in label_to_indices:
                label_to_indices[agg_label] = []
                label_to_areas[agg_label] = []
            label_to_indices[agg_label].append(row['Parcel Index'])
            label_to_areas[agg_label].append(row['Area Name'])

        # Create new atlas with combined regions
        new_atlas = np.zeros_like(atlas_data)
        new_labels = []
        next_label = 1

        for agg_label, indices in label_to_indices.items():
            # Create mask for all regions with this aggregate label
            mask = np.zeros_like(atlas_data, dtype=bool)
            for idx in indices:
                mask |= (atlas_data == idx)

            # Assign new label to all voxels in the combined region
            new_atlas[mask] = next_label

            # Add to new labels
            new_labels.append({
                'Parcel Index': next_label,
                'Aggregate Label': agg_label,
                'Area Names': ', '.join(sorted(set(label_to_areas[agg_label])))
            })

            next_label += 1

        atlas_data = new_atlas
        labels = pd.DataFrame(new_labels)

        return atlas_data, labels

    elif atlas_format == 'volume':
        from nilearn.image import resample_to_img

        atlas = nib.load(os.path.join(data_dir, 'glasser', 'MNI_Glasser_HCP_v1.0.nii.gz'))

        if nifti_fn is not None:
            atlas = resample_to_img(atlas, nifti_fn, interpolation='nearest')

        # Filter atlas data to only include requested regions if needed
        if regions is not None:
            data = atlas.get_fdata()
            mask = np.isin(data, labels['Parcel Index'])
            data[~mask] = 0
            atlas = nib.Nifti1Image(data, atlas.affine, atlas.header)

        # Create mapping from aggregate labels to parcel indices and area names
        label_to_indices = {}
        label_to_areas = {}
        for _, row in labels.iterrows():
            agg_label = row[aggregate_column]
            if agg_label not in label_to_indices:
                label_to_indices[agg_label] = []
                label_to_areas[agg_label] = []
            label_to_indices[agg_label].append(row['Parcel Index'])
            label_to_areas[agg_label].append(row['Area Name'])

        # Create new atlas with combined regions
        data = atlas.get_fdata()
        new_data = np.zeros_like(data)
        new_labels = []
        next_label = 1

        for agg_label, indices in label_to_indices.items():
            # Create mask for all regions with this aggregate label
            mask = np.zeros_like(data, dtype=bool)
            for idx in indices:
                mask |= (data == idx)

            # Assign new label to all voxels in the combined region
            new_data[mask] = next_label

            # Add to new labels
            new_labels.append({
                'Parcel Index': next_label,
                aggregate_column: agg_label,
                'Area Names': ', '.join(sorted(set(label_to_areas[agg_label])))
            })

            next_label += 1

        atlas = nib.Nifti1Image(new_data, atlas.affine, atlas.header)
        labels = pd.DataFrame(new_labels)

        return atlas, labels

    else:
        raise ValueError("atlas_format must be either 'surface' or 'volume'")

# Atlas Manipulation Functions
def create_atlas_mask(atlas, labels, label_type='Aggregate Label', selected_labels=None):
    """Create a mask volume/surface with only the selected labels.

    Args:
        atlas (array-like or Nifti1Image): Atlas data, either as array for surface or Nifti image for volume
        labels (pd.DataFrame): DataFrame containing label mappings with columns 'Aggregate Label' and 'Parcel Index'
        label_type (str): Column name in labels DataFrame to use for filtering
        selected_labels (list): List of label names to include in mask

    Returns:
        array-like or Nifti1Image: Binary mask with 1s for selected regions and 0s elsewhere
    """
    # Get indices for selected labels
    all_indices = []

    assert label_type in labels.columns, f"Label type {label_type} not found in labels"

    idx = [labels[labels[label_type] == label]['Parcel Index'].values
            for label in selected_labels]
    idx = np.concatenate(idx)

    # Add indices for both hemispheres
    all_indices.extend(idx)
    all_indices.extend(idx + 1000)
    all_indices = np.array(all_indices)

    # Create binary mask
    if isinstance(atlas, nib.Nifti1Image):
        # For volume atlas
        atlas_data = atlas.get_fdata()
        mask = np.isin(atlas_data, all_indices).astype(float)
        mask = nib.Nifti1Image(mask, atlas.affine, atlas.header)
    else:
        # For surface atlas
        mask = np.isin(atlas, all_indices).astype(float)

    return mask

def data_to_parcel(data, atlas, labels, mask=None, coverage_threshold=0.5, average_parcel=True):
    """Convert voxel-wise/vertex-wise data to parcels using an atlas.

    Args:
        data (array-like): Data to convert to parcels (1D array for both surface and volume)
        atlas (array-like): Atlas data with parcel indices (1D array for surface, 3D for volume)
        labels (array-like): Label names corresponding to parcel indices
        mask (array-like, optional): Binary mask to restrict analysis
        coverage_threshold (float): Minimum proportion of voxels/vertices needed in a parcel
        average_parcel (bool): Whether to average values within parcels

    Returns:
        pd.DataFrame: DataFrame with parcel-wise data and metadata
    """
    # Convert volume atlas to 1D if needed
    if len(atlas.shape) > 1:
        atlas = atlas.reshape(-1)

    unique_labels = np.unique(labels)
    parcel_data = []

    for label in unique_labels:
        # Get vertices/voxels for this parcel
        parcel_mask = (atlas == label) | (atlas == (label + 1000))  # Account for both hemispheres

        # If mask is provided, only include voxels/vertices that are in the mask
        if mask is not None:

            # Mask the mask image by the parcel and calculate coverage
            mask_parcel = mask[parcel_mask]
            coverage = np.sum(mask_parcel) / len(mask_parcel)

            # Find the intersection of the parcel mask and the mask
            parcel_mask = np.logical_and(parcel_mask, mask)
            parcel_values = data[parcel_mask]
        else:
            parcel_values = data[parcel_mask]
            coverage = 1

        if coverage >= coverage_threshold and len(parcel_values) > 0:
            if average_parcel:
                value = np.nanmean(parcel_values)
            else:
                value = parcel_values

            parcel_data.append({
                'label': label,
                'value': value,
                'coverage': coverage
            })

    return pd.DataFrame(parcel_data)

def combine_atlases(atlas_list, atlas_names=None, atlas_labels=None, overlap_threshold=0.5, atlas_priority=None):
    """Combine multiple atlases into a single atlas, handling overlaps based on priority.

    In overlapping areas, the atlas with higher priority takes precedence. Non-overlapping
    parts of regions are preserved even if they overlap with higher priority regions elsewhere.

    Args:
        atlas_list (list): List of numpy arrays, each representing an atlas
        atlas_names (list, optional): List of names for each atlas. If None, uses 'atlas_0', 'atlas_1', etc.
        atlas_labels (list, optional): List of label DataFrames for each atlas. Each DataFrame should have
            'Parcel Index' and 'Aggregate Label' columns. If None, only numeric indices will be used.
        overlap_threshold (float, optional): Threshold for considering regions as overlapping. Default is 0.5.
        atlas_priority (list, optional): List of atlas names in order of priority. Earlier atlases take precedence
            over later ones in overlapping regions. If None, no priority is applied.

    Returns:
        tuple: (combined_atlas, parcel_info)
            - combined_atlas: numpy array where each region gets a unique label
            - parcel_info: pandas.DataFrame with columns:
                - Parcel Index: unique label in combined atlas
                - Area Name: region name
                - Aggregate Label: atlas name
                - Overlapping Regions: list of overlapping region names
    """
    # Input validation
    if atlas_names is None:
        atlas_names = [f'atlas_{i}' for i in range(len(atlas_list))]

    if len(atlas_list) != len(atlas_names):
        raise ValueError("Number of atlases must match number of atlas names")

    if atlas_labels is not None and len(atlas_list) != len(atlas_labels):
        raise ValueError("Number of atlases must match number of label DataFrames")

    # Set up priority mapping
    if atlas_priority is not None:
        missing_atlases = set(atlas_priority) - set(atlas_names)
        if missing_atlases:
            raise ValueError(f"Priority list contains unknown atlases: {missing_atlases}")
        priority_map = {name: idx for idx, name in enumerate(atlas_priority)}
    else:
        priority_map = {name: idx for idx, name in enumerate(atlas_names)}

    # Find all regions and their overlaps
    regions, overlaps = _find_overlapping_regions(
        atlas_list, atlas_names, atlas_labels, overlap_threshold
    )

    # Create combined atlas
    combined_atlas = np.zeros_like(atlas_list[0])
    parcel_info = []
    next_label = 1

    # Track which voxels have been assigned
    assigned_voxels = np.zeros_like(combined_atlas, dtype=bool)

    # Process regions in priority order
    for region_label, (atlas_name, label) in sorted(regions.items(),
                                                   key=lambda x: priority_map.get(x[1][0], float('inf'))):
        # Get region mask
        atlas_idx = atlas_names.index(atlas_name)
        region_mask = (atlas_list[atlas_idx] == label)

        # Only assign label to voxels that haven't been assigned yet
        unassigned_mask = ~assigned_voxels & region_mask
        if np.any(unassigned_mask):
            combined_atlas[unassigned_mask] = next_label

            # Add to parcel info
            parcel_info.append({
                'Parcel Index': next_label,
                'Area Name': region_label,
                'Aggregate Label': atlas_name,
                'Overlapping Regions': ', '.join(overlaps[region_label]) if overlaps[region_label] else 'None'
            })

            # Mark these voxels as assigned
            assigned_voxels[unassigned_mask] = True

            next_label += 1

    return combined_atlas, pd.DataFrame(parcel_info)

def get_atlas_difference_regions(atlas_list, atlas_names=None, atlas_labels=None):
    """Get a dictionary of regions unique to each atlas.

    Args:
        atlas_list (list): List of numpy arrays, each representing an atlas
        atlas_names (list, optional): List of names for each atlas. If None, uses 'atlas_0', 'atlas_1', etc.
        atlas_labels (list, optional): List of label DataFrames for each atlas. Each DataFrame should have
            'Parcel Index' and 'Aggregate Label' columns. If None, only numeric indices will be used.

    Returns:
        tuple: (unique_regions, parcel_info)
            - unique_regions: dict mapping atlas names to lists of their unique region labels
            - parcel_info: pandas.DataFrame with parcel information
    """
    combined_atlas, parcel_info = combine_atlases(
        atlas_list, atlas_names, atlas_labels
    )

    # Create dictionary of unique regions per atlas
    unique_regions = {name: [] for name in atlas_names}

    # For each atlas
    for atlas_idx, (atlas, atlas_name) in enumerate(zip(atlas_list, atlas_names)):
        # Get unique labels in this atlas
        unique_labels = np.unique(atlas)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        # For each unique label in this atlas
        for label in unique_labels:
            # Create mask for this region
            region_mask = (atlas == label)

            # Get overlapping regions in combined atlas
            overlapping_regions = combined_atlas[region_mask]
            overlapping_labels = np.unique(overlapping_regions[overlapping_regions != 0])

            # If this region is labeled with this atlas's number in the combined atlas
            if len(overlapping_labels) == 1 and overlapping_labels[0] == atlas_idx + 1:
                unique_regions[atlas_name].append(int(label))

    return unique_regions, parcel_info
