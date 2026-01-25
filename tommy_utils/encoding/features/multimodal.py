"""Multimodal feature extraction for encoding models."""

import os
import numpy as np
import torch
from tqdm import trange

from ...config.models import MULTIMODAL_MODELS_DICT, MULTIMODAL_CONFIG
from ..utils.helpers import get_device, chunk_video_clips


def load_multimodal_model(model_name, cache_dir=None):
    """Load a multimodal model ford feature extraction.

    Parameters
    ----------
    model_name : str
        Model name from MULTIMODAL_MODELS_DICT (e.g., 'clip', 'pe_av')
    cache_dir : str, optional
        Cache directory for model files

    Returns
    -------
    processor : transformers.AutoProcessor
        Model processor
    model : transformers.PreTrainedModel
        Pretrained model

    Raises
    ------
    ValueError
        If model_name is not in MULTIMODAL_MODELS_DICT
    """
    if cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

    from transformers import AutoProcessor, AutoModel

    if model_name not in MULTIMODAL_MODELS_DICT:
        raise ValueError(f'Model {model_name} not in MULTIMODAL_MODELS_DICT')

    # For some reason loading model after processor causes nans with video embedding model
    model = AutoModel.from_pretrained(MULTIMODAL_MODELS_DICT[model_name]) #, use_safetensors=True)
    processor = AutoProcessor.from_pretrained(MULTIMODAL_MODELS_DICT[model_name])

    model.eval()
    
    return processor, model


def create_multimodal_features(
    model_name,
    video=None,
    audio=None,
    images=None,
    text=None,
    batch_size=8,
    normalize=True,
    device=None,
    cache_dir=None,
    window_size=2.0,
    temporal_subsample=None,
):
    """Create multimodal features from various inputs.

    Parameters
    ----------
    model_name : str
        Model name ('clip', 'pe_av')
    video : VideoDecoder, optional
    audio : AudioDecoder, optional
    images : array-like or list, optional
    text : str or list of str, optional
    batch_size : int, default=8
    normalize : bool, default=True
        L2-normalize embeddings
    device : str or torch.device, optional
    cache_dir : str, optional
    window_size : float, default=2.0
        For video/audio_video: Duration of temporal context window in seconds
        Not used for text or static images.
    temporal_subsample : float, optional
        For video/audio_video: Time interval between feature extractions in seconds.
        If None (default), extracts features for every frame.
        Not used for text or static images.

    Returns
    -------
    times : np.ndarray or None
        Timestamps (None for non-temporal data)
    features : np.ndarray
        Extracted features

    Examples
    --------
    >>> create_multimodal_features('clip', text=['Hello'])
    >>> create_multimodal_features('clip', video=decoder)
    >>> create_multimodal_features('clip', images=[img1, img2])
    >>> # Extract features for every frame using 2s of context (default)
    >>> create_multimodal_features('pe_av', video=vid_dec, audio=aud_dec, window_size=2.0)
    >>> # Extract features every 0.5s using 2s of context
    >>> create_multimodal_features('pe_av', video=vid_dec, audio=aud_dec, window_size=2.0, temporal_subsample=0.5)
    >>> # Extract features for every frame using 1s of context
    >>> create_multimodal_features('pe_av', video=vid_dec, audio=aud_dec, window_size=1.0)
    """
    # Validate model and infer modality from inputs
    config = MULTIMODAL_CONFIG[model_name]

    if video and audio and 'audio_video' in config['modalities']:
        modality = 'audio_video'
    elif text and 'text' in config['modalities']:
        modality = 'text'
    elif (video or images) and 'image' in config['modalities']:
        modality = 'image'
    else:
        raise ValueError(f'Invalid inputs for {model_name}. Supported: {config["modalities"]}')

    # Setup
    device = get_device(device)
    processor, model = load_multimodal_model(model_name, cache_dir)
    model = model.to(device)
    extract_fn = getattr(model, config['methods'][modality])

    features_list = []
    times = None

    # TEXT
    if modality == 'text':
        text = [text] if isinstance(text, str) else text
        inputs = processor(text=text, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            embeds = extract_fn(**inputs).detach().cpu()
            if normalize:
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        features_list.append(embeds.numpy())

    # IMAGE
    elif modality == 'image':
        if video:
            fps = int(video.metadata.average_fps)
            times = np.arange(len(video)) / fps
            batches = chunk_video_clips(video, batch_size)
        else:
            images = [images] if not isinstance(images, (list, tuple)) else images
            batches = (images[i:i + batch_size] for i in range(0, len(images), batch_size))

        for batch in batches:
            inputs = processor(images=batch, return_tensors='pt')
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                embeds = extract_fn(**inputs).detach().cpu()
                if normalize:
                    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            features_list.append(embeds.numpy())

    # AUDIO-VIDEO
    else:
        # Get video metadata
        metadata = video.metadata
        video_fps = metadata.average_fps
        total_duration = metadata.duration_seconds_from_header
        n_frames = len(video)

        # Create time points based on temporal_subsample
        if temporal_subsample is None:
            # Extract for every frame
            times = np.arange(n_frames) / video_fps
        else:
            # Extract at temporal_subsample intervals
            assert temporal_subsample <= window_size, 'Temporal subsample must be less than or equal to window size'
            times = np.arange(0, total_duration, temporal_subsample)

        # Get required sample rate for the model
        source_sr = audio.metadata.sample_rate
        # For PE_AV and similar models, check processor config for required sample rate
        if hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            target_sr = processor.feature_extractor.sampling_rate
        else:
            # Default to 48000 for PE_AV if not found in processor
            target_sr = 48000 if 'pe_av' in model_name.lower() else source_sr

        # Resample audio decoder if needed
        if source_sr != target_sr:
            print(f"Setting audio from {source_sr} Hz to {target_sr} Hz")
            audio._desired_sample_rate = target_sr

        # Calculate the number of samples/frames needed for a full window
        window_audio_samples = int(window_size * target_sr)
        window_video_frames = int(window_size * video_fps)

        # Use trange for progress bar
        iterator = trange(len(times), desc=f'Extracting {model_name} features')

        for idx in iterator:
            current_time = times[idx]

            # Calculate temporal window: [current_time - window_size, current_time]
            window_start_time = max(0, current_time - window_size)
            if temporal_subsample is None:
                window_end_time = current_time + (1 / video_fps)  # Include current frame
            else:
                window_end_time = max(current_time, temporal_subsample)  # Ensure non-zero end time

            # Extract video frames in the temporal window
            video_samples = video.get_frames_played_in_range(window_start_time, window_end_time)

            # Extract audio for the same temporal window
            audio_samples = audio.get_samples_played_in_range(window_start_time, window_end_time)

            # Verify resampling on first iteration
            if idx == 0:
                actual_sr = audio_samples.sample_rate
                if actual_sr != target_sr:
                    raise RuntimeError(
                        f"Audio resampling failed: expected {target_sr} Hz but got {actual_sr} Hz"
                    )

            audio_data = audio_samples.data.squeeze(0)
            video_data = video_samples.data

            # Convert stereo to mono if needed (average channels)
            if audio_data.ndim > 1 and audio_data.shape[0] > 1:
                audio_data = audio_data.mean(dim=0, keepdim=False)

            # Pad audio with zeros at the beginning if we don't have enough samples
            if audio_data.shape[-1] < window_audio_samples:
                pad_size = window_audio_samples - audio_data.shape[-1]
                audio_data = torch.nn.functional.pad(audio_data, (pad_size, 0), mode='constant', value=0)

            # Pad video with zeros at the beginning if we don't have enough frames
            # video_data is [T, C, H, W]
            if video_data.shape[0] < window_video_frames:
                pad_size = window_video_frames - video_data.shape[0]
                # Pad temporal dimension (dim 0)
                video_data = torch.nn.functional.pad(
                    video_data, (0, 0, 0, 0, 0, 0, pad_size, 0), mode='constant', value=0
                )

            inputs = processor(
                videos=video_data,
                audio=audio_data,
                fps=video_fps,
                sampling_rate=target_sr,
                return_tensors='pt',
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                embeds = extract_fn(**inputs)
                embeds = embeds.audio_video_embeds.detach().cpu().float()
            features_list.append(embeds.numpy())

    return times, np.vstack(features_list)
