"""Model configuration for tommy_utils.

This module contains dictionaries mapping model names to their identifiers
or download paths for various feature extraction models.
"""

# Encoding feature extractors available by modality
ENCODING_FEATURES = {
    'visual': [
        'alexnet',
        'clip',
        'motion_energy',
    ],
    'audio': [
        'spectral',
    ],
    'language': [
        'phoneme',
        'word2vec',
        'gpt2',
        'gpt2-xl',
    ]
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

# Multimodal models (e.g., CLIP)
MULTIMODAL_MODELS_DICT = {
    'clip': "openai/clip-vit-base-patch32"
}
