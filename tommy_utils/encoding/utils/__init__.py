"""Encoding utilities subpackage.

This subpackage provides:
- Helper functions for data handling and feature preparation
- Cross-validation/validation strategies
- Model evaluation and metrics
- Model I/O (save/load)
"""

# Helper functions
from .helpers import (
    get_modality_features,
    load_gentle_transcript,
    create_banded_features,
    get_concatenated_data,
    get_train_test_splits,
    lanczosinterp2D,
    lanczosfun
)

# Validation
from .validation import generate_leave_one_run_out

# Evaluation
from .evaluation import (
    get_all_banded_metrics,
    BANDED_RIDGE_MODELS,
    KERNEL_RIDGE_MODELS
)

# I/O
from .io import (
    save_model_parameters,
    load_model_from_parameters
)

__all__ = [
    # Helpers
    'get_modality_features',
    'load_gentle_transcript',
    'create_banded_features',
    'get_concatenated_data',
    'get_train_test_splits',
    'lanczosinterp2D',
    'lanczosfun',
    # Validation
    'generate_leave_one_run_out',
    # Evaluation
    'get_all_banded_metrics',
    'BANDED_RIDGE_MODELS',
    'KERNEL_RIDGE_MODELS',
    # I/O
    'save_model_parameters',
    'load_model_from_parameters',
]
