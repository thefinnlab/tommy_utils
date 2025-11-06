"""Audio feature extraction for encoding models.

This module provides functions for extracting audio features:
- Mel-spectrograms
- Spectral features
- TorchAudio model features
"""

import numpy as np
import torch
import torchaudio


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
