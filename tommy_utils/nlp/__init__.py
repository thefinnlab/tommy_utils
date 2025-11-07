"""NLP utilities for natural language processing.

This subpackage provides:
- Word embeddings (GloVe, Word2Vec, FastText)
- Transformer models (GPT, BERT, etc.)
- Contextualized word embeddings
- Word prediction and semantic analysis
"""

# Import everything from nlp for now (to be refactored later)
from .nlp import *

__all__ = [
    # Word embeddings
    'load_word_model',
    'get_basis_vector',
    'get_word_score',
    'get_word_clusters',
    'find_word_clusters',
    'autovivify_list',
    # Transformer models
    'load_clm_model',
    'load_mlm_model',
    'load_multimodal_model',
    # Inference
    'get_clm_predictions',
    # Text processing
    'get_segment_indices',
    'transcript_to_input',
    # Embedding extraction
    'get_word_prob',
    'subwords_to_words',
    'extract_word_embeddings',
    # Analysis
    'create_results_dataframe',
    'get_word_vector_metrics',
    'get_model_statistics',
]
