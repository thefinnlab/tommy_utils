"""Feature extraction subpackage for encoding models.

This module uses lazy imports to avoid loading heavy dependencies
(torch, transformers, etc.) until they're actually needed.
"""

# Type hints for IDE support
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .language import (
        create_phoneme_features,
        create_word_features,
        create_transformer_features
    )
    from .vision import (
        create_vision_features,
        extract_image_features,
        extract_video_features,
        extract_multimodal_features,
        create_motion_energy_features,
        frames_to_luminance,
        load_torchvision_model,
        VISION_MODELS_DICT,
        PYMOTEN_DEFAULT_PARAMS
    )
    from .audio import (
        create_spectral_features,
        load_torchaudio_model
    )


def __getattr__(name):
    """Lazy load feature extraction functions."""
    # Language features
    if name in ('create_phoneme_features', 'create_word_features',
                'create_transformer_features'):
        from .language import (
            create_phoneme_features,
            create_word_features,
            create_transformer_features
        )
        return locals()[name]

    # Vision features
    elif name in ('create_vision_features', 'extract_image_features',
                  'extract_video_features', 'extract_multimodal_features',
                  'create_motion_energy_features', 'frames_to_luminance',
                  'load_torchvision_model', 'VISION_MODELS_DICT',
                  'PYMOTEN_DEFAULT_PARAMS'):
        from .vision import (
            create_vision_features,
            extract_image_features,
            extract_video_features,
            extract_multimodal_features,
            create_motion_energy_features,
            frames_to_luminance,
            load_torchvision_model,
            VISION_MODELS_DICT,
            PYMOTEN_DEFAULT_PARAMS
        )
        return locals()[name]

    # Audio features
    elif name in ('create_spectral_features', 'load_torchaudio_model'):
        from .audio import (
            create_spectral_features,
            load_torchaudio_model
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Language features
    'create_phoneme_features',
    'create_word_features',
    'create_transformer_features',
    # Vision features
    'create_vision_features',
    'extract_image_features',
    'extract_video_features',
    'extract_multimodal_features',
    'create_motion_energy_features',
    'frames_to_luminance',
    'load_torchvision_model',
    'VISION_MODELS_DICT',
    'PYMOTEN_DEFAULT_PARAMS',
    # Audio features
    'create_spectral_features',
    'load_torchaudio_model',
]
