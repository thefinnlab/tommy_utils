"""
HTTPie: modern, user-friendly command-line HTTP client for the API era.

"""

from . import (
	afni,
	delayer, 
	encoding, 
	fmriprep, 
	nlp, 
	plotting,
	statistics
)

__version__ = '0.0.1'
__date__ = '2024-02-07'
__author__ = 'Tommy Botch'

__all__ = [
	'delayer',
	'encoding',
	'fmriprep',
	'nlp',
	'plotting',
	'statistics'
]