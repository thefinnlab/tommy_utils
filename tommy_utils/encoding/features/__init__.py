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
        create_motion_energy_features,
        create_scene_cut_features,
        load_torchvision_model,
        VISION_MODELS_DICT,
        PYMOTEN_DEFAULT_PARAMS
    )
    from .audio import (
        create_spectral_features,
        create_audio_features,
        load_torchaudio_model,
        load_audio_model
    )
    from .multimodal import (
        create_multimodal_features,
        load_multimodal_model
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
                  'extract_video_features', 'create_motion_energy_features',
                  'create_scene_cut_features', 'load_torchvision_model',
                  'VISION_MODELS_DICT', 'PYMOTEN_DEFAULT_PARAMS'):
        from .vision import (
            create_vision_features,
            extract_image_features,
            extract_video_features,
            create_motion_energy_features,
            create_scene_cut_features,
            load_torchvision_model,
            VISION_MODELS_DICT,
            PYMOTEN_DEFAULT_PARAMS
        )
        return locals()[name]

    # Audio features
    elif name in ('create_spectral_features', 'create_audio_features',
                  'load_torchaudio_model', 'load_audio_model'):
        from .audio import (
            create_spectral_features,
            create_audio_features,
            load_torchaudio_model,
            load_audio_model
        )
        return locals()[name]

    # Multimodal features
    elif name in ('create_multimodal_features', 'load_multimodal_model'):
        from .multimodal import (
            create_multimodal_features,
            load_multimodal_model
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
    'create_motion_energy_features',
    'create_scene_cut_features',
    'load_torchvision_model',
    'VISION_MODELS_DICT',
    'PYMOTEN_DEFAULT_PARAMS',
    # Audio features
    'create_spectral_features',
    'create_audio_features',
    'load_torchaudio_model',
    'load_audio_model',
    # Multimodal features
    'create_multimodal_features',
    'load_multimodal_model',
]
