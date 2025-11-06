"""Utilities for extracting confounds from fMRIPrep outputs.

This module provides functions for working with fMRIPrep confound files,
including extraction of CompCor components (aCompCor and tCompCor) and
flexible confound selection.

Thanks to Sam Nastase for the original code.
"""

import os
import json
import pandas as pd
from natsort import natsorted


def extract_compcor(confounds_df, confounds_meta, n_comps=5, method='tCompCor',
                   tissue=None):
    """Extract CompCor components from fMRIPrep confounds.

    Parameters
    ----------
    confounds_df : pd.DataFrame
        fMRIPrep confounds dataframe
    confounds_meta : dict
        fMRIPrep confounds metadata dictionary from JSON sidecar
    n_comps : int or float, default=5
        If >= 1: Number of components to extract
        If < 1: Proportion of variance to explain
    method : str, default='tCompCor'
        CompCor method to use. Options: 'aCompCor', 'tCompCor'
    tissue : str or None, default=None
        Required for aCompCor. Options: 'combined', 'CSF', 'WM'
        Ignored for tCompCor

    Returns
    -------
    pd.DataFrame
        DataFrame containing the selected CompCor components

    Raises
    ------
    ValueError
        If invalid parameters are provided
    """
    # Validate inputs
    if n_comps <= 0:
        raise ValueError("n_comps must be greater than 0")

    if method not in ['aCompCor', 'tCompCor']:
        raise ValueError("method must be 'aCompCor' or 'tCompCor'")

    if method == 'aCompCor' and tissue not in ['combined', 'CSF', 'WM']:
        raise ValueError(
            "Must specify a tissue type (combined, CSF, or WM) for aCompCor"
        )

    # Warn if tissue specified for tCompCor
    if method == 'tCompCor' and tissue:
        print(f"Warning: tCompCor is not restricted to a tissue mask - "
              f"ignoring tissue specification ({tissue})")
        tissue = None

    # Get CompCor metadata for relevant method
    compcor_meta = {
        c: confounds_meta[c] for c in confounds_meta
        if confounds_meta[c]['Method'] == method
        and confounds_meta[c]['Retained']
    }

    # If aCompCor, filter metadata for tissue mask
    if method == 'aCompCor':
        compcor_meta = {
            c: compcor_meta[c] for c in compcor_meta
            if compcor_meta[c]['Mask'] == tissue
        }

    # Make sure metadata components are sorted properly by singular value
    comp_sorted = natsorted(compcor_meta)
    for i, comp in enumerate(comp_sorted[:-1]):
        comp_next = comp_sorted[i + 1]
        if not (compcor_meta[comp]['SingularValue'] >
                compcor_meta[comp_next]['SingularValue']):
            raise ValueError(
                f"Components not sorted by singular value: "
                f"{comp} vs {comp_next}"
            )

    # Get top n components
    if n_comps >= 1.0:
        n_comps = int(n_comps)
        if len(comp_sorted) >= n_comps:
            comp_selector = comp_sorted[:n_comps]
        else:
            comp_selector = comp_sorted
            print(f"Warning: Only {len(comp_sorted)} {method} "
                  f"components available ({n_comps} requested)")

    # Or get components necessary to capture n proportion of variance
    else:
        comp_selector = []
        for comp in comp_sorted:
            comp_selector.append(comp)
            if compcor_meta[comp]['CumulativeVarianceExplained'] > n_comps:
                break

    # Ensure at least one component was selected
    if len(comp_selector) == 0:
        raise ValueError("No components selected - check your inputs")

    # Extract the actual component time series
    confounds_compcor = confounds_df[comp_selector]

    return confounds_compcor


def extract_group(confounds_df, groups):
    """Extract confound groups (e.g., motion outliers, cosines) from confounds.

    Parameters
    ----------
    confounds_df : pd.DataFrame
        fMRIPrep confounds dataframe
    groups : str or list of str
        Group label(s) to extract (e.g., 'cosine', 'motion_outlier')

    Returns
    -------
    pd.DataFrame
        DataFrame containing all columns matching the group labels
    """
    # Ensure groups is a list
    if isinstance(groups, str):
        groups = [groups]

    # Filter for all columns containing any of the group labels
    confounds_group = []
    for group in groups:
        group_cols = [col for col in confounds_df.columns if group in col]
        if group_cols:
            confounds_group.append(confounds_df[group_cols])

    if not confounds_group:
        return pd.DataFrame()

    confounds_group = pd.concat(confounds_group, axis=1)

    return confounds_group


def load_confounds(confounds_fn):
    """Load fMRIPrep confounds TSV and JSON sidecar.

    Parameters
    ----------
    confounds_fn : str
        Path to the confounds TSV file

    Returns
    -------
    confounds_df : pd.DataFrame
        Confounds dataframe
    confounds_meta : dict
        Confounds metadata from JSON sidecar
    """
    # Load the confounds TSV file
    confounds_df = pd.read_csv(confounds_fn, sep='\t')

    # Load the JSON sidecar metadata
    json_fn = os.path.splitext(confounds_fn)[0] + '.json'
    with open(json_fn) as f:
        confounds_meta = json.load(f)

    return confounds_df, confounds_meta


def extract_confounds(confounds_df, confounds_meta, model_spec):
    """Extract confounds based on a model specification.

    Parameters
    ----------
    confounds_df : pd.DataFrame
        fMRIPrep confounds dataframe
    confounds_meta : dict
        fMRIPrep confounds metadata dictionary
    model_spec : dict
        Model specification dictionary. Should contain:
        - 'confounds': list of confound column names
        - 'aCompCor': (optional) dict or list of dicts with CompCor params
        - 'tCompCor': (optional) dict or list of dicts with CompCor params

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested confounds

    Examples
    --------
    >>> model_spec = {
    ...     'confounds': ['trans_x', 'trans_y', 'trans_z',
    ...                   'rot_x', 'rot_y', 'rot_z', 'cosine'],
    ...     'aCompCor': {'n_comps': 5, 'tissue': 'CSF'}
    ... }
    >>> confounds = extract_confounds(confounds_df, confounds_meta, model_spec)
    """
    # Identify confound groups (variable number of columns)
    groups = set(model_spec['confounds']).intersection(
        ['cosine', 'motion_outlier']
    )

    # Grab the requested individual confounds
    individual_confounds = [c for c in model_spec['confounds'] if c not in groups]
    confounds = confounds_df[individual_confounds] if individual_confounds else pd.DataFrame()

    # Grab confound groups if present
    if groups:
        confounds_group = extract_group(confounds_df, list(groups))
        if not confounds.empty and not confounds_group.empty:
            confounds = pd.concat([confounds, confounds_group], axis=1)
        elif not confounds_group.empty:
            confounds = confounds_group

    # Get aCompCor / tCompCor confounds if requested
    compcors = set(model_spec).intersection(['aCompCor', 'tCompCor'])
    if compcors:
        for compcor in compcors:
            # Ensure compcor params is a list of dicts
            compcor_params = model_spec[compcor]
            if isinstance(compcor_params, dict):
                compcor_params = [compcor_params]

            for compcor_kws in compcor_params:
                confounds_compcor = extract_compcor(
                    confounds_df,
                    confounds_meta,
                    method=compcor,
                    **compcor_kws
                )

                if not confounds.empty:
                    confounds = pd.concat([confounds, confounds_compcor], axis=1)
                else:
                    confounds = confounds_compcor

    return confounds
