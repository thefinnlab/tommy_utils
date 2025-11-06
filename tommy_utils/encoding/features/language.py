"""Language feature extraction for encoding models.

This module provides functions for extracting language features including:
- Phoneme features from forced alignment
- Word embeddings (Word2Vec, GloVe, FastText)
- Contextualized embeddings from transformer models
"""

import numpy as np
import pandas as pd
from ...config.paths import get_phonemes_path


# Load CMU phoneme dictionary
CMU_PHONEMES = pd.read_csv(
    get_phonemes_path(),
    header=None,
    names=['phoneme', 'type'],
    sep="\t"
)


def create_phoneme_features(df_transcript):
    """Convert Gentle transcript to one-hot phoneme features.

    Parameters
    ----------
    df_transcript : pd.DataFrame
        Gentle transcript with phoneme alignments

    Returns
    -------
    times : np.ndarray
        Onset times for each phoneme
    phoneme_features : np.ndarray
        One-hot encoded phonemes (39 dimensions)
    """
    phoneme_time_features = []

    for i, row in df_transcript.iterrows():
        if row['case'] != 'success' or row['alignedWord'] == '<unk>':
            continue

        phoneme_start = row['start']
        word_phonemes = []

        for item in row['phones']:
            phoneme = item['phone'].split('_')[0].upper()
            one_hot_phoneme = np.asarray(CMU_PHONEMES['phoneme'] == phoneme).astype(int)

            if sum(one_hot_phoneme) != 1:
                print(f'Word {i} - skipping phoneme: {phoneme}')
                phoneme_start += item['duration']
                continue

            phoneme_info = (phoneme_start, one_hot_phoneme)
            word_phonemes.append(phoneme_info)
            phoneme_start += item['duration']

        phoneme_time_features.append(word_phonemes)

    phoneme_time_features = sum(phoneme_time_features, [])
    times, phoneme_features = [np.stack(item) for item in zip(*phoneme_time_features)]

    print(f'Phoneme feature space is size: {phoneme_features.shape}')

    return times, phoneme_features


def create_word_features(df_transcript, word_model):
    """Extract word embeddings from Gensim model.

    Parameters
    ----------
    df_transcript : pd.DataFrame
        Gentle transcript
    word_model : gensim.models
        Word embedding model (Word2Vec, FastText, etc.)

    Returns
    -------
    times : np.ndarray
        Word onset times
    word_features : np.ndarray
        Word embedding vectors
    """
    word_time_features = []

    for i, row in df_transcript.iterrows():
        word = row['word']

        # Check if word exists in model vocabulary
        if 'fasttext' in str(type(word_model)) or word in word_model.key_to_index:
            word_vector = word_model[word]
        else:
            continue

        word_time_info = (row['start'], word_vector)
        word_time_features.append(word_time_info)

    times, word_features = [np.stack(item) for item in zip(*word_time_features)]

    print(f'Word feature space is size: {word_features.shape}')

    return times, word_features


def create_transformer_features(df_transcript, tokenizer, model, window_size=25,
                                bidirectional=False, add_punctuation=False):
    """Extract contextualized word embeddings from transformer models.

    Parameters
    ----------
    df_transcript : pd.DataFrame
        Gentle transcript
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model
    model : transformers.PreTrainedModel
        Transformer model
    window_size : int
        Context window size for contextualization
    bidirectional : bool
        Whether to use bidirectional context
    add_punctuation : bool
        Whether to include punctuation in context

    Returns
    -------
    times : np.ndarray
        Word onset times
    word_features : np.ndarray
        Contextualized embeddings (n_layers, n_words, hidden_size)
    """
    # Import nlp module functions at runtime to avoid circular dependency
    from ... import nlp

    word_time_features = []

    # Create segments for windowed contextualization
    segments = nlp.get_segment_indices(
        n_words=len(df_transcript),
        window_size=window_size,
        bidirectional=bidirectional
    )

    for (i, row), segment in zip(df_transcript.iterrows(), segments):
        print(f'Processing segment {i+1}/{len(df_transcript)}')

        inputs = nlp.transcript_to_input(
            df_transcript,
            segment,
            add_punctuation=add_punctuation
        )

        # Extract embeddings for the last word (target word)
        word_embeddings = nlp.extract_word_embeddings(
            [inputs],
            tokenizer,
            model,
            word_indices=-1
        )
        word_embeddings = word_embeddings.squeeze()

        word_time_info = (row['start'], word_embeddings.squeeze())
        word_time_features.append(word_time_info)

    times, word_features = [np.stack(item) for item in zip(*word_time_features)]

    # Move layers to first dimension: (n_layers, n_words, hidden_size)
    word_features = np.moveaxis(word_features, 0, 1)

    print(f'Transformer feature space is size: {word_features.shape}')

    return times, word_features
