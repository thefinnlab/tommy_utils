"""Utilities for creating AFNI regressors.

This module provides functions for creating different types of regressors
for AFNI's 3dDeconvolve, including amplitude-modulated and duration-modulated
regressors.
"""

import pandas as pd


def create_stim_times_regressor(df, onset_var, timing_offset=0):
    """Create a simple stimulus times regressor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stimulus information
    onset_var : str
        Column name for onset times
    timing_offset : float, default=0
        Offset to add to onset times (in seconds)

    Returns
    -------
    df : pd.DataFrame
        DataFrame after filtering NaNs
    regressor : list of str
        List of onset times formatted for AFNI
    """
    print(f'There are {len(df)} entries')

    df = df.dropna()
    print(f'There are {len(df)} entries after removing NaNs')

    # Apply timing offset
    df[onset_var] = df[onset_var] + timing_offset

    regressor = [f'{onset:.2f}' for onset in df[onset_var].values]
    return df, regressor


def create_AM_regressor(df, AM_var, onset_var, timing_offset=0):
    """Create an amplitude-modulated regressor.

    Pairs event onset times with amplitude values for parametric modulation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stimulus information
    AM_var : str
        Column name for amplitude values
    onset_var : str
        Column name for onset times
    timing_offset : float, default=0
        Offset to add to onset times (in seconds)

    Returns
    -------
    df : pd.DataFrame
        DataFrame after filtering NaNs
    regressor : list of str
        List of onset*amplitude pairs formatted for AFNI
    """
    print(f'There are {len(df)} entries')

    df = df.dropna()
    print(f'There are {len(df)} entries after removing NaNs')

    # Apply timing offset
    df[onset_var] = df[onset_var] + timing_offset
    regressor = [f'{onset:.2f}*{amplitude:.2f}'
                 for onset, amplitude in df[[onset_var, AM_var]].values]

    return df, regressor


def create_DM_regressor(df, DM_var, onset_var, timing_offset=0, min_duration=0):
    """Create a duration-modulated regressor.

    Pairs event onset times with duration values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stimulus information
    DM_var : str
        Column name for duration values
    onset_var : str
        Column name for onset times
    timing_offset : float, default=0
        Offset to add to onset times (in seconds)
    min_duration : float, default=0
        Minimum duration threshold (in seconds). Events shorter than this
        will be excluded.

    Returns
    -------
    df : pd.DataFrame
        DataFrame after filtering (includes 'time_filter' column)
    regressor : list of str
        List of onset:duration pairs formatted for AFNI
    """
    print(f'There are {len(df)} entries')

    df = df.dropna()
    print(f'There are {len(df)} entries after removing NaNs')

    # Apply timing offset
    df[onset_var] = df[onset_var] + timing_offset

    # Filter by duration
    time_filter = df[DM_var] >= min_duration
    df['time_filter'] = time_filter

    print(f'{sum(time_filter)} entries are longer than {min_duration} seconds')

    filtered = df[df['time_filter']]
    regressor = [f'{onset:.2f}:{duration:.2f}'
                 for onset, duration in filtered[[onset_var, DM_var]].values]
    return df, regressor


def create_DM_AM_regressor(df, DM_var, AM_var, onset_var, timing_offset=0,
                          min_duration=0):
    """Create a duration-modulated and amplitude-modulated regressor.

    Combines duration and amplitude modulation: onset*amplitude:duration

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing stimulus information
    DM_var : str
        Column name for duration values
    AM_var : str
        Column name for amplitude values
    onset_var : str
        Column name for onset times
    timing_offset : float, default=0
        Offset to add to onset times (in seconds)
    min_duration : float, default=0
        Minimum duration threshold (in seconds). Events shorter than this
        will be excluded.

    Returns
    -------
    df : pd.DataFrame
        DataFrame after filtering (includes 'time_filter' column)
    regressor : list of str
        List of onset*amplitude:duration triplets formatted for AFNI
    """
    print(f'There are {len(df)} entries')

    df = df.dropna()
    print(f'There are {len(df)} entries after removing NaNs')

    # Apply timing offset
    df[onset_var] = df[onset_var] + timing_offset

    # Filter by duration
    time_filter = df[DM_var] >= min_duration
    df['time_filter'] = time_filter

    print(f'{sum(time_filter)} entries are longer than {min_duration} seconds')

    filtered = df[df['time_filter']]
    regressor = [f'{onset:.2f}*{amplitude:.2f}:{duration:.2f}'
                 for onset, duration, amplitude
                 in filtered[[onset_var, DM_var, AM_var]].values]
    return df, regressor


def write_regressor(regressor_list, out_fn):
    """Write regressor list to file.

    Parameters
    ----------
    regressor_list : list of str
        List of formatted regressor values
    out_fn : str
        Output file path
    """
    with open(out_fn, 'w') as f:
        f.write(' '.join(regressor_list))
