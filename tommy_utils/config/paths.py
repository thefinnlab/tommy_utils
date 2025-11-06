"""Path configuration for tommy_utils data files."""

import os
from pathlib import Path


def get_data_dir():
    """Get the path to the data directory.

    Returns
    -------
    Path
        Path to the tommy_utils/data directory
    """
    return Path(__file__).parent.parent / 'data'


def get_phonemes_path():
    """Get the path to the CMU phoneme dictionary.

    Returns
    -------
    Path
        Path to cmudict-0.7b.phones.txt
    """
    return get_data_dir() / 'nlp' / 'cmudict-0.7b.phones.txt'


def get_atlas_dir():
    """Get the path to the atlas data directory.

    Returns
    -------
    Path
        Path to the atlases directory
    """
    return get_data_dir() / 'atlases'


def get_nlp_data_dir():
    """Get the path to the NLP data directory.

    Returns
    -------
    Path
        Path to the NLP data directory
    """
    return get_data_dir() / 'nlp'
