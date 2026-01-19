#!/usr/bin/env python
"""
Tommy's Neuroscience Utilities

This package provides utilities for fMRI analysis and encoding model development.

Major components:
- config: Centralized configuration for models and paths
- encoding: Feature extraction, modeling pipelines, and custom solvers
- fmri: fMRI preprocessing (fMRIPrep, AFNI, atlases)
- nlp: Natural language processing and transformers
- visualization: Brain visualization and statistical plots
- stats: Statistical testing for neuroimaging
"""

# Import refactored subpackages
from . import config
from . import encoding
from . import fmri
from . import nlp
from . import visualization

# Import other modules
from . import (
	decomp,
	stats,
	misc
)

# Backward compatibility: expose submodules at top level
from .fmri import afni, atlas, fmriprep
from .encoding import delayer
from .encoding import solvers as custom_solvers

__version__ = '0.2.0'
__date__ = '2025-01-06'
__author__ = 'Tommy Botch'

__all__ = [
	# Refactored subpackages
	'config',
	'encoding',
	'fmri',
	'nlp',
	'visualization',
	# Other modules
	'decomp',
	'stats',
	'misc',
	# Backward compatibility: top-level access to submodules
	'afni',
	'atlas',
	'delayer',
	'fmriprep',
	'custom_solvers',
]
