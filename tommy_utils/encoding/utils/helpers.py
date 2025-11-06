"""Utility functions for encoding models."""

import json
import numpy as np
import pandas as pd
from operator import itemgetter
from ...config.models import ENCODING_FEATURES


def get_modality_features(modality):
    """Get available feature extractors for a given modality.

    Parameters
    ----------
    modality : str
        One of 'audiovisual', 'audio', 'text', or 'visual'

    Returns
    -------
    list
        List of available feature extractor names
    """
    modality_map = {
        'audiovisual': ['visual', 'audio', 'language'],
        'audio': ['audio', 'language'],
        'text': ['language'],
        'visual': ['visual']
    }

    items = modality_map.get(modality, [])
    modality_features = []

    for item in items:
        if ENCODING_FEATURES.get(item):
            modality_features.extend(ENCODING_FEATURES[item])

    return modality_features


def load_gentle_transcript(transcript_fn, start_offset=None):
    """Load and process a Gentle alignment transcript.

    Parameters
    ----------
    transcript_fn : str
        Path to Gentle JSON transcript file
    start_offset : float, optional
        Time offset to apply to all timestamps

    Returns
    -------
    pd.DataFrame
        Transcript with columns: word, start, end, punctuation
    """
    with open(transcript_fn) as f:
        data = json.load(f)

    transcript = data['transcript']
    df_transcript = pd.json_normalize(data['words'])

    for i, row in df_transcript.iterrows():
        # get the punctuation of the current row
        if i+1 < len(df_transcript):
            start_punc, end_punc = row['endOffset'], df_transcript.loc[i+1, 'startOffset']
            word_punctuation = transcript[start_punc:end_punc]
        else:
            word_punctuation = transcript[row['endOffset']:]

        df_transcript.loc[i, 'punctuation'] = word_punctuation

    # Interpolate missing times
    df_transcript['start'] = df_transcript['start'].interpolate()
    df_transcript['word'] = df_transcript.word.str.lower()

    # Apply time offset if provided
    if start_offset:
        df_transcript['start'] -= start_offset
        df_transcript['end'] -= start_offset

    return df_transcript


def create_banded_features(features, feature_names):
    """Prepare features for banded ridge regression.

    Parameters
    ----------
    features : list of np.ndarray
        List of feature arrays for different feature spaces
    feature_names : list of str
        Names for each feature space

    Returns
    -------
    features : np.ndarray
        Concatenated features across all feature spaces
    feature_space_info : list of tuple
        List of (name, slice) pairs for each feature space
    """
    features_dim = [feature.shape[1] for feature in features]

    # Create slices for each feature space
    feature_space_idxs = np.concatenate([[0], np.cumsum(features_dim)])
    feature_space_slices = [
        slice(*item) for item in zip(feature_space_idxs[:-1], feature_space_idxs[1:])
    ]

    assert len(feature_space_slices) == len(feature_names)

    # Concatenate feature spaces horizontally
    features = np.concatenate(features, axis=1)

    # Pair names with slices
    feature_space_info = [
        (name, slice_) for name, slice_ in zip(feature_names, feature_space_slices)
    ]

    return features, feature_space_info


def get_concatenated_data(data, indices, precision='float32'):
    """Concatenate data from specified indices.

    Parameters
    ----------
    data : list of np.ndarray
        List of data arrays
    indices : list of int
        Indices to concatenate
    precision : str
        Data type precision

    Returns
    -------
    np.ndarray
        Concatenated and cleaned data
    """
    if len(indices) > 1:
        data_split = np.concatenate(
            itemgetter(*indices)(data), axis=0
        ).astype(precision)
    else:
        data_split = np.stack(
            itemgetter(*indices)(data), axis=0
        ).astype(precision)

    # Convert nan to num
    data_split = np.nan_to_num(data_split)

    # Convert inf to num
    data_split[np.isinf(data_split)] = 0

    return data_split


def get_train_test_splits(x, y, train_indices, test_indices,
                          precision='float32', group_level=False):
    """Get train and test data splits.

    Parameters
    ----------
    x : list of np.ndarray
        Feature arrays
    y : list of np.ndarray
        Target arrays
    train_indices : list of int
        Training indices
    test_indices : list of int
        Test indices
    precision : str
        Data type precision
    group_level : bool
        Whether using group-level modeling

    Returns
    -------
    X_train, Y_train, X_test, Y_test : np.ndarray
        Training and test splits
    """
    # Get train data
    if group_level:
        assert (len(x) == 1)
        X_train = get_concatenated_data(x, [0], precision)
        X_test = get_concatenated_data(x, [0], precision)
    else:
        X_train = get_concatenated_data(x, train_indices, precision)
        X_test = get_concatenated_data(x, test_indices, precision)

    # Get test data
    Y_train = get_concatenated_data(y, train_indices, precision)
    Y_test = get_concatenated_data(y, test_indices, precision)

    return X_train, Y_train, X_test, Y_test


def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolate data using Lanczos resampling.

    Adapted from Huth Lab:
    https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/interpdata.py

    Parameters
    ----------
    data : np.ndarray
        Data to interpolate (rows = timepoints, columns = features)
    oldtime : np.ndarray
        Original time points
    newtime : np.ndarray
        Target time points (evenly spaced)
    window : int
        Number of lobes in sinc function
    cutoff_mult : float
        Cutoff frequency multiplier
    rectify : bool
        Whether to rectify positive and negative components separately

    Returns
    -------
    newdata : np.ndarray
        Interpolated data at new time points
    """
    # Calculate cutoff frequency from target sampling rate
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult

    # Build sinc interpolation matrix
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)

    if rectify:
        # Interpolate positive and negative components separately
        newdata = np.hstack([
            np.dot(sincmat, np.clip(data, -np.inf, 0)),
            np.dot(sincmat, np.clip(data, 0, np.inf))
        ])
    else:
        newdata = np.dot(sincmat, data)

    return newdata


def lanczosfun(cutoff, t, window=3):
    """Compute windowed sinc (Lanczos) function.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency
    t : float or np.ndarray
        Time points
    window : int
        Number of lobes (window size)

    Returns
    -------
    val : float or np.ndarray
        Lanczos function values
    """
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0
    return val
