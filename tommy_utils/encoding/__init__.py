"""Encoding model subpackage for fMRI encoding analysis.

This subpackage provides functionality for:
- Feature extraction (vision, audio, language)
- Building encoding pipelines
- Custom solvers for Himalaya
- HRF delay modeling
- Utilities (validation, evaluation, I/O)
"""

# Import submodules
from . import solvers
from . import delayer
from . import utils

# Feature extraction
from .features import (
    create_phoneme_features,
    create_word_features,
    create_transformer_features,
    create_vision_features,
    create_spectral_features,
    load_torchvision_model,
    load_torchaudio_model,
    VISION_MODELS_DICT
)

# Re-export config model dicts for convenience
from ..config.models import (
    ENCODING_FEATURES,
    WORD_MODELS,
    CLM_MODELS_DICT,
    MLM_MODELS_DICT,
    MULTIMODAL_MODELS_DICT
)

# Utilities (expose at encoding level for convenience)
from .utils import (
    get_modality_features,
    load_gentle_transcript,
    create_banded_features,
    get_concatenated_data,
    get_train_test_splits,
    lanczosinterp2D,
    lanczosfun,
    generate_leave_one_run_out,
    get_all_banded_metrics,
    BANDED_RIDGE_MODELS,
    KERNEL_RIDGE_MODELS,
    save_model_parameters,
    load_model_from_parameters
)

# Pipeline building
from .pipeline import (
    create_banded_model,
    build_encoding_pipeline
)

__all__ = [
    # Submodules
    'solvers',
    'delayer',
    'utils',
    # Model configuration dicts
    'ENCODING_FEATURES',
    'WORD_MODELS',
    'CLM_MODELS_DICT',
    'MLM_MODELS_DICT',
    'MULTIMODAL_MODELS_DICT',
    'VISION_MODELS_DICT',
    # Feature extraction - language
    'create_phoneme_features',
    'create_word_features',
    'create_transformer_features',
    # Feature extraction - vision
    'create_vision_features',
    'load_torchvision_model',
    # Feature extraction - audio
    'create_spectral_features',
    'load_torchaudio_model',
    # Utilities
    'get_modality_features',
    'load_gentle_transcript',
    'create_banded_features',
    'get_concatenated_data',
    'get_train_test_splits',
    'lanczosinterp2D',
    'lanczosfun',
    # Validation
    'generate_leave_one_run_out',
    # Pipeline
    'create_banded_model',
    'build_encoding_pipeline',
    # Evaluation
    'get_all_banded_metrics',
    'BANDED_RIDGE_MODELS',
    'KERNEL_RIDGE_MODELS',
    # I/O
    'save_model_parameters',
    'load_model_from_parameters',
]
