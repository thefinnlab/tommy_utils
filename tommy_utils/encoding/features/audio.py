"""Audio feature extraction for encoding models.

This module provides functions for extracting audio features:
- Mel-spectrograms
- Spectral features
- TorchAudio model features
- Transformer-based audio features (Wav2Vec2, HuBERT)
"""

import os
import numpy as np
import torch
import torchaudio
from tqdm import trange


def load_torchaudio_model(model_name):
    """Load a TorchAudio pre-trained model.

    Parameters
    ----------
    model_name : str
        Name of the TorchAudio bundle (uppercased)

    Returns
    -------
    bundle : torchaudio.pipelines bundle
        Model bundle with configuration
    model : torch.nn.Module
        Pre-trained model
    """
    bundle = getattr(torchaudio.pipelines, model_name.upper())
    model = bundle.get_model()
    return bundle, model


def create_spectral_features(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Create mel-spectrogram features from audio.

    Parameters
    ----------
    audio : torch.Tensor
        Audio waveform
    sr : int
        Sample rate
    n_fft : int
        FFT window size
    hop_length : int
        Hop length between frames
    n_mels : int
        Number of mel frequency bins

    Returns
    -------
    times : np.ndarray
        Time points for each spectral frame
    melspec : np.ndarray
        Mel-spectrogram features (n_frames, n_mels)
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        center=True,
        n_fft=n_fft,
        hop_length=hop_length,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(audio).squeeze()

    # Create evenly-spaced time points for spectrogram frames
    times = np.linspace(0, audio.shape[-1]/sr, melspec.shape[-1])

    # Transpose to (n_frames, n_mels)
    return times, melspec.T


def load_audio_model(model_name, cache_dir=None):
    """Load an audio transformer model.

    Parameters
    ----------
    model_name : str
        Model name ('wav2vec2', 'hubert')
    cache_dir : str, optional
        Cache directory for model files

    Returns
    -------
    processor : transformers.AutoProcessor
        Model processor
    model : transformers.PreTrainedModel
        Pretrained model
    """
    if cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

    from transformers import AutoProcessor, AutoModel
    from ...config import AUDIO_MODELS_DICT

    if model_name not in AUDIO_MODELS_DICT:
        raise ValueError(f'Model {model_name} not supported. Available: {list(AUDIO_MODELS_DICT.keys())}')

    model_path = AUDIO_MODELS_DICT[model_name]
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    return processor, model


def create_audio_features(
    model_name,
    audio,
    window_size=2.0,
    temporal_subsample=0.25,
    pool=True,
    device=None,
    cache_dir=None,
):
    """Create audio features using transformer models.

    Parameters
    ----------
    model_name : str
        Model name ('wav2vec2', 'hubert')
    audio : AudioDecoder
        torchcodec AudioDecoder
    window_size : float, default=2.0
        Duration of temporal context window in seconds
    temporal_subsample : float, default=0.25
        Time interval between feature extractions in seconds
    pool : bool, default=True
        Whether to mean-pool features over each window
    device : str or torch.device, optional
    cache_dir : str, optional

    Returns
    -------
    times : np.ndarray
        Time points for each feature
    features : np.ndarray
        Extracted features

    Examples
    --------
    >>> from torchcodec.decoders import AudioDecoder
    >>> audio = AudioDecoder('audio.wav', sample_rate=16000)
    >>> # Extract features every 0.5s using 2s of context (default)
    >>> times, features = create_audio_features('wav2vec2', audio)
    >>> # Extract features every 0.25s using 1s of context
    >>> times, features = create_audio_features('wav2vec2', audio, window_size=1.0, temporal_subsample=0.25)
    """
    from ..utils.helpers import get_device

    device = get_device(device)

    assert temporal_subsample <= window_size, 'Temporal subsample must be less than or equal to window size'

    # Load model
    processor, model = load_audio_model(model_name, cache_dir)
    model = model.to(device)

    # Get audio properties
    total_duration = audio.metadata.duration_seconds_from_header
    source_sr = audio.metadata.sample_rate
    target_sr = processor.feature_extractor.sampling_rate

    # Resample audio decoder if needed by adjusting desired sample rate
    if source_sr != target_sr:
        print(f"Setting audio from {source_sr} Hz to {target_sr} Hz")
        audio._desired_sample_rate = target_sr

    # Calculate the number of samples needed for a full window
    window_samples = int(window_size * target_sr)

    # Create time points at temporal_subsample intervals
    times = np.arange(0, total_duration, temporal_subsample)
    features = []

    # Process each time window with sliding window approach
    for idx in trange(len(times), desc=f'Extracting {model_name} features'):
        current_time = times[idx]

        # Calculate temporal window: [current_time - window_size, current_time]
        # This provides window_size seconds of backward-looking context
        window_start_time = max(0, current_time - window_size)
        window_end_time = max(current_time, temporal_subsample)  # Ensure non-zero end time

        # Extract audio samples in the temporal window (decoder handles resampling)
        audio_samples = audio.get_samples_played_in_range(window_start_time, window_end_time)

        # Verify resampling worked correctly
        if idx == 0:  # Check on first iteration
            actual_sr = audio_samples.sample_rate
            if actual_sr != target_sr:
                raise RuntimeError(
                    f"Audio resampling failed: expected {target_sr} Hz but got {actual_sr} Hz"
                )

        waveform = audio_samples.data.squeeze(0)

        # Pad with zeros at the beginning if we don't have enough samples for a full window
        if waveform.shape[-1] < window_samples:
            pad_size = window_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (pad_size, 0), mode='constant', value=0)

        inputs = processor(waveform.numpy(), sampling_rate=target_sr,
                          return_tensors='pt', return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state

            if pool:
                # Mean pool over time
                attention_mask = model._get_feature_vector_attention_mask(
                    embeds.shape[1], inputs['attention_mask']
                )
                embeds = (embeds * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
                embeds = embeds.squeeze(0)  # Remove batch dim
            else:
                embeds = embeds.squeeze(0)  # Just remove batch dim

            embeds = embeds.detach().cpu()

        features.append(embeds.numpy())

    # Stack features
    features = np.vstack(features)

    # For unpooled features, times may not align perfectly with features
    # since each window produces variable-length sequences
    if not pool:
        # Recalculate times based on actual feature count
        frame_rate = 50.0  # Approximate frame rate for wav2vec2/hubert
        times = np.arange(len(features)) / frame_rate

    return times, features
