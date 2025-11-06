"""fMRI preprocessing and analysis utilities.

This subpackage provides:
- fMRIPrep confound extraction
- AFNI regressor creation
- Brain atlas manipulation and parcellation
"""

from . import fmriprep
from . import afni
from . import atlas

__all__ = ['fmriprep', 'afni', 'atlas']
