"""Model configuration for tommy_utils.

This module contains dictionaries mapping model names to their identifiers
or download paths for various feature extraction models.
"""

# Encoding feature extractors available by modality
ENCODING_FEATURES = {
    'visual': [
        'alexnet',
        'x3d_s',
        'motion_energy',
        'scene_cut',
    ],
    'audio': [
        'spectral',
        'wav2vec2',
    ],
    'language': [
        'phoneme',
        'word2vec',
        'gpt2',
        'gpt2-xl',
    ],
    'multimodal': [
        'clip',
        'pe_av',
    ],
}

# Word embedding models (GloVe, Word2Vec, FastText)
WORD_MODELS = {
    'glove': 'glove.42B.300d.zip',
    'word2vec': 'word2vec-google-news-300',
    'fasttext': 'cc.en.300.bin'
}

# Causal Language Models (CLMs) - autoregressive models like GPT
CLM_MODELS_DICT = {
    'bloom': 'bigscience/bloom-560m',
    'gpt2': 'gpt2',
    'gpt2-xl': 'gpt2-xl',
    'gpt-neo-x': 'EleutherAI/gpt-neo-1.3B',
    'llama2': 'meta-llama/Llama-2-7b-hf',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'qwen3-8B': 'Qwen/Qwen3-8B',
    'qwen3-32B': 'Qwen/Qwen3-32B',
    'llama3.1-8B': 'meta-llama/Llama-3.1-8B',
    'llama3.1-70B': 'meta-llama/Llama-3.1-70B',
    'gemma3-1b-pt': 'google/gemma-3-1b-pt'
}

# Masked Language Models (MLMs) - bidirectional models like BERT
MLM_MODELS_DICT = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'electra': 'google/electra-base-generator',
    'xlm-prophetnet': 'microsoft/xprophetnet-large-wiki100-cased'
}

# Audio transformer models (Wav2Vec2, HuBERT)
AUDIO_MODELS_DICT = {
    'wav2vec2': 'facebook/wav2vec2-large-960h-lv60',
    'hubert': 'facebook/hubert-large-ls960',
}

# Multimodal models (e.g., CLIP)
MULTIMODAL_MODELS_DICT = {
    'clip': "openai/clip-vit-base-patch32",
    'pe_av': "facebook/pe-av-base-16-frame",  # placeholder - update with actual model path
}

# Multimodal model configuration
MULTIMODAL_CONFIG = {
    'clip': {
        'modalities': ['text', 'image'],
        'methods': {'text': 'get_text_features', 'image': 'get_image_features'},
    },
    'pe_av': {
        'modalities': ['audio_video'],
        'methods': {'audio_video': 'get_audio_video_embeds'},
    },
}

# Vision model layer configurations for feature extraction
VISION_MODELS_DICT = {
    'alexnet': [
        'maxpool2d_1_3',
        'maxpool2d_2_6',
        'relu_3_8',
        'relu_4_10',
        'maxpool2d_3_13',
        'relu_6_18',
        'relu_7_21'
    ],
    'resnet50': [
        'layer1.2.relu:3',
        'layer2.3.relu:3',
        'layer3.5.relu:3',
        'layer4.2.relu:3'
    ],
    # X3D temporal video model
    'x3d_s': {
        'type': 'temporal',
        'side_size': 182,
        'crop_size': 182,
        'num_frames': 13,
        'sampling_rate': 6,
        'feature_dim': 2048,
        'layers': [
            'blocks.1.res_blocks.2.activation',
            'blocks.2.res_blocks.4.activation',
            'blocks.3.res_blocks.10.activation',
            'blocks.4.res_blocks.6.activation',
        ]
    }
}

# Default parameters for pymoten motion energy extraction
PYMOTEN_DEFAULT_PARAMS = {
    # Pyramid filter configuration
    'temporal_frequencies': [0, 2, 4],              # Hz
    'spatial_frequencies': [0, 2, 4, 8, 16],        # cycles per image
    'spatial_directions': [0, 45, 90, 135, 180, 225, 270, 315],  # degrees

    # Filter shape parameters
    'filter_temporal_width': 'auto',                # frames or 'auto' (~2/3 fps)
    'sf_gauss_ratio': 0.6,                          # spatial frequency Gaussian ratio
    'max_spatial_env': 0.3,                         # maximum spatial envelope
    'filter_spacing': 3.5,                          # spacing between filters
    'tf_gauss_ratio': 10.0,                         # temporal frequency Gaussian ratio
    'max_temp_env': 0.3,                            # maximum temporal envelope
    'include_edges': False,                         # include edge filters
    'spatial_phase_offset': 0.0,                    # filter phase offset (degrees)

    # Processing parameters
    'downsample_factor': None,                      # downsample factor (e.g., 2 = downsample by 2x) or None
}
