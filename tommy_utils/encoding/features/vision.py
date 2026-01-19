"""Vision feature extraction for encoding models.

This module provides functions for extracting vision features from images/video:
- Pre-trained CNN features (AlexNet, ResNet50)
- X3D temporal video features
- Motion energy features (pymoten)
- Scene cut detection (PySceneDetect)
- Layer-wise feature extraction
"""

import numpy as np
import torch
import torchvision
import torchlens as tl
from torchvision import transforms
from tqdm import trange

from ...config.models import VISION_MODELS_DICT, PYMOTEN_DEFAULT_PARAMS
from ..utils.helpers import get_device, chunk_video_clips


def load_torchvision_model(model_name):
    """Load a pre-trained torchvision model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'alexnet', 'resnet50')

    Returns
    -------
    model : torch.nn.Module
        Pre-trained model
    """
    return getattr(torchvision.models, model_name)(pretrained=True)


def get_layer_tensors(model_output, flatten=True):
    """Extract layer activations from torchlens output.

    Parameters
    ----------
    model_output : torchlens.ModelHistory
        Output from torchlens.log_forward_pass
    flatten : bool
        Whether to flatten spatial dimensions

    Returns
    -------
    layer_tensors : list of torch.Tensor
        Extracted layer activations
    """
    layer_tensors = []
    for layer in model_output.layers_with_saved_activations:
        arr = model_output[layer].tensor_contents.detach()
        layer_tensors.append(torch.flatten(arr, 1, -1) if flatten else arr)
    return layer_tensors


def _get_standard_transform():
    """Get standard ImageNet preprocessing transform."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_layer_features(model, batch, model_layers, transform, device):
    """Extract multi-layer activations using torchlens (works with any PyTorch model)."""
    batch = torch.stack([transform(x) for x in batch]).to(device)
    model_output = tl.log_forward_pass(model, batch, layers_to_save=model_layers)
    return get_layer_tensors(model_output)


def _get_x3d_transform(num_frames, side_size, crop_size):
    """Get X3D-specific preprocessing transform using pytorchvideo pipeline.

    Parameters
    ----------
    num_frames : int
        Number of frames to subsample to
    side_size : int
        Size for short side scaling
    crop_size : int
        Size for center crop

    Returns
    -------
    transform : ApplyTransformToKey
        Transform that expects dictionary with 'video' key
    """
    from torchvision.transforms import Compose, Lambda
    from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]),
    )


def _reduce_temporal_features(tensor, reduction='last'):
    """Reduce temporal dimension of video features.

    Parameters
    ----------
    tensor : torch.Tensor
        Feature tensor with shape [B, C, T, H, W]
    reduction : str or int
        How to reduce temporal dimension:
        - 'last': Use last frame (index -1)
        - 'first': Use first frame (index 0)
        - 'mean': Global average pooling across T, H, W
        - int: Use specific frame index

    Returns
    -------
    torch.Tensor
        Reduced tensor with shape [B, C] for 'mean' or [B, C, H, W] for frame selection
    """
    if reduction == 'mean':
        # Global average pooling: [B, C, T, H, W] -> [B, C]
        return tensor.mean(dim=[2, 3, 4])
    elif reduction == 'last':
        # Last frame: [B, C, T, H, W] -> [B, C, H, W]
        return tensor[:, :, -1, :, :]
    elif reduction == 'first':
        # First frame: [B, C, T, H, W] -> [B, C, H, W]
        return tensor[:, :, 0, :, :]
    elif isinstance(reduction, int):
        # Specific frame index
        return tensor[:, :, reduction, :, :]
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'last', 'first', or int.")


def extract_image_features(images, model_name, batch_size=8, verbose=True, device=None):
    """Extract frame-by-frame multi-layer features (CNNs, Vision Transformers, etc.).

    Parameters
    ----------
    images : video decoder
        Video frame decoder
    model_name : str
        Model name ('alexnet', 'resnet50', or any torchvision model)
    batch_size : int
        Batch size for processing frames
    verbose : bool
        Whether to print progress
    device : str or torch.device, optional
        Device to use. If None, automatically selects 'cuda' if available.

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    features : list of np.ndarray
        Multi-layer features, one array per layer
    """
    device = get_device(device)
    model = load_torchvision_model(model_name)
    model = model.to(device)
    model.eval()
    model_layers = VISION_MODELS_DICT[model_name]
    transform = _get_standard_transform()

    video_fps = int(images.metadata.average_fps)
    n_frames = len(images)
    times = np.arange(0, n_frames / video_fps, 1/video_fps)

    if verbose:
        print(f'Using device: {device}')

    vision_features = []
    for i, batch in enumerate(chunk_video_clips(images, batch_size=batch_size)):
        if verbose:
            print(f'Processing batch {i+1}/{n_frames // batch_size}')
        vision_features.append(_extract_layer_features(model, batch, model_layers, transform, device))

    vision_features = [np.vstack(item) for item in zip(*vision_features)]
    if verbose:
        print(f'Image feature space has {len(vision_features)} layers')

    return times, vision_features


def extract_video_features(images, model_name, window_size=2.0, temporal_subsample=None, verbose=True, device=None, **kwargs):
    """Extract temporal video features with context (X3D).

    Parameters
    ----------
    images : video decoder
        Video frame decoder with get_frames_played_in_range() support
    model_name : str
        Video model name ('x3d_s', 'x3d_xs', 'x3d_m')
    window_size : float, default=2.0
        Duration of temporal context window in seconds (e.g., 2.0 = use 2s of video for each feature)
    temporal_subsample : float, optional
        Time interval between feature extractions in seconds (e.g., 0.5 = extract features every 0.5s).
        If None (default), extracts features for every frame.
    verbose : bool
        Whether to print progress
    device : str or torch.device, optional
        Device to use. If None, automatically selects 'cuda' if available.
    **kwargs : dict
        sampling_rate : int, optional
            Temporal frame sampling interval (overrides model default)
        temporal_reduction : str or int, default='last'
            How to reduce temporal dimension of layer activations:
            - 'last': Use last frame (index -1)
            - 'first': Use first frame (index 0)
            - 'mean': Global average pooling across T, H, W -> [B, C]
            - int: Use specific frame index
        flatten : bool, default=True
            Whether to flatten spatial dimensions after temporal reduction

    Returns
    -------
    times : np.ndarray
        Time points in seconds for each feature
    features : list of np.ndarray
        Multi-layer video features, one array per layer [n_timepoints, feature_dim]

    Notes
    -----
    This function extracts features using a sliding window approach:
    - Features are extracted for every frame (if temporal_subsample=None) or at regular intervals (if temporal_subsample is specified)
    - Each feature uses window_size seconds of backward-looking temporal context
    - Adjacent windows overlap, providing temporal continuity
    - Temporal reduction collapses the model's internal temporal dimension to one vector per timepoint

    Examples
    --------
    >>> # Extract features for every frame using 2s of context (default)
    >>> times, features = extract_video_features(decoder, 'x3d_s')

    >>> # Extract features every 0.5s using 2s of context
    >>> times, features = extract_video_features(decoder, 'x3d_s', window_size=2.0, temporal_subsample=0.5)

    >>> # Extract features for every frame using 1s of context
    >>> times, features = extract_video_features(decoder, 'x3d_s', window_size=1.0)

    >>> # Extract features with global average pooling
    >>> times, features = extract_video_features(
    ...     decoder, 'x3d_s', window_size=2.0, temporal_reduction='mean'
    ... )
    """
    device = get_device(device)

    # Get model configuration
    model_config = VISION_MODELS_DICT[model_name]
    num_frames = model_config['num_frames']
    side_size = model_config['side_size']
    crop_size = model_config['crop_size']
    model_layers = model_config['layers']
    sampling_rate = kwargs.get('sampling_rate', model_config['sampling_rate'])
    temporal_reduction = kwargs.get('temporal_reduction', 'last')
    flatten = kwargs.get('flatten', True)

    # Get video metadata
    metadata = images.metadata
    video_fps = metadata.average_fps
    total_duration = metadata.duration_seconds_from_header
    n_frames = len(images)

    # Calculate the number of frames needed for a full window
    window_frames = int(window_size * video_fps)

    # Create time points based on temporal_subsample
    if temporal_subsample is None:
        # Extract for every frame
        times = np.arange(n_frames) / video_fps
    else:
        # Extract at temporal_subsample intervals
        assert temporal_subsample <= window_size, 'Temporal subsample must be less than or equal to window size'
        times = np.arange(0, total_duration, temporal_subsample)

    if verbose:
        print(f'Using device: {device}')
        print(f'Video model: {model_name}')
        print(f'  Frames per clip: {num_frames}')
        print(f'  Sampling rate: {sampling_rate} ({video_fps/sampling_rate:.1f} fps effective)')
        print(f'  Window size: {window_size}s (temporal context)')
        if temporal_subsample is None:
            print(f'  Extracting features for every frame ({video_fps:.1f} fps)')
        else:
            print(f'  Temporal subsample: {temporal_subsample}s (extraction interval)')
            overlap_pct = max(0, (window_size - temporal_subsample) / window_size * 100) if window_size > 0 else 0
            print(f'  Window overlap: {overlap_pct:.1f}%')
        print(f'  Processing {len(times)} timepoints from 0 to {total_duration:.2f}s')
        print(f'  Temporal reduction: {temporal_reduction}')

    # Load model
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model = model.to(device)
    model.eval()

    # Create transform
    transform = _get_x3d_transform(num_frames, side_size, crop_size)

    # Process video with sliding window at specified intervals
    all_layer_features = [[] for _ in model_layers]

    # Use trange for progress bar if verbose, otherwise regular range
    iterator = trange(len(times), desc=f'Extracting {model_name} features', disable=not verbose) if verbose else range(len(times))

    for idx in iterator:
        current_time = times[idx]

        # Calculate temporal window: [current_time - window_size, current_time]
        # This provides window_size seconds of backward-looking context
        window_start_time = max(0, current_time - window_size)
        if temporal_subsample is None:
            window_end_time = current_time + (1 / video_fps)  # Include current frame
        else:
            window_end_time = max(current_time, temporal_subsample)  # Ensure non-zero end time

        # Extract frames in the temporal window using time-based method
        frames = images.get_frames_played_in_range(window_start_time, window_end_time)

        # frames.data is [T, C, H, W], convert to [C, T, H, W]
        video_tensor = frames.data.permute(1, 0, 2, 3)

        # Pad with zeros at the beginning if we don't have enough frames for a full window
        if video_tensor.shape[1] < window_frames:
            pad_size = window_frames - video_tensor.shape[1]
            # Pad temporal dimension: (left, right) for last dim, then work backwards
            # video_tensor is [C, T, H, W], so pad T (dim 1) requires padding format for last 2 dims first
            video_tensor = torch.nn.functional.pad(
                video_tensor, (0, 0, 0, 0, pad_size, 0), mode='constant', value=0
            )

        # Create dictionary format expected by ApplyTransformToKey
        video_data = {"video": video_tensor}

        # Apply transform (includes UniformTemporalSubsample)
        video_data = transform(video_data)
        inputs = video_data["video"].unsqueeze(0).to(device)  # [1, C, T, H, W]

        # Extract features from all layers
        with torch.no_grad():
            model_output = tl.log_forward_pass(
                model,
                inputs,
                layers_to_save=model_layers
            )

        # Process each layer's activations
        for layer_idx, layer in enumerate(model_output.layers_with_saved_activations):
            # Get raw tensor: [B, C, T, H, W]
            arr = model_output[layer].tensor_contents.detach()

            # Apply temporal reduction
            reduced = _reduce_temporal_features(arr, reduction=temporal_reduction)

            # Optionally flatten spatial dimensions
            if flatten and reduced.ndim > 2:
                reduced = torch.flatten(reduced, 1, -1)
            
            # Store features
            all_layer_features[layer_idx].append(reduced.cpu().numpy())

    # Stack features for each layer: list of [n_frames, feature_dim]
    vision_features = [np.vstack(layer_feats) for layer_feats in all_layer_features]

    if verbose:
        print(f'\nExtracted features from {len(model_layers)} layers:')
        for i, (layer_name, feats) in enumerate(zip(model_layers, vision_features)):
            print(f'  Layer {i+1} ({layer_name}): {feats.shape}')

    return times, vision_features


def create_vision_features(images, model_name, batch_size=8, verbose=True, device=None, window_size=2.0, temporal_subsample=None, **kwargs):
    """Extract vision features from video frames.

    Dispatcher function that routes to specialized extraction functions based on model type.

    Parameters
    ----------
    images : video decoder
        Video frame decoder
    model_name : str
        Name of vision model ('alexnet', 'resnet50', 'clip', 'x3d_s')
    batch_size : int
        Batch size for processing (not used for X3D models)
    verbose : bool
        Whether to print progress
    device : str or torch.device, optional
        Device to use. If None, automatically selects 'cuda' if available.
    window_size : float, default=2.0
        For X3D models: Duration of temporal context window in seconds
        Not used for other models.
    temporal_subsample : float, optional
        For X3D models: Time interval between feature extractions in seconds.
        If None (default), extracts features for every frame.
        Not used for other models.
    **kwargs : dict
        Model-specific parameters (passed to specialized functions).
        For X3D models:
            sampling_rate : int, optional
                Temporal frame sampling interval (overrides model default)
            temporal_reduction : str or int, default='last'
                How to reduce temporal dimension ('last', 'first', 'mean', or int)
            flatten : bool, default=True
                Whether to flatten spatial dimensions after temporal reduction

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    vision_features : np.ndarray or list of np.ndarray
        Vision features (single array for CLIP/X3D, list of arrays for layer-wise CNN models)

    Examples
    --------
    >>> # Frame-based CNN
    >>> times, features = create_vision_features(decoder, 'alexnet')

    >>> # CLIP with GPU
    >>> times, features = create_vision_features(decoder, 'clip', device='cuda')

    >>> # X3D with features for every frame using 2s of context (default)
    >>> times, features = create_vision_features(decoder, 'x3d_s', window_size=2.0)

    >>> # X3D with features every 0.5s using 2s of context
    >>> times, features = create_vision_features(decoder, 'x3d_s', window_size=2.0, temporal_subsample=0.5)

    >>> # X3D with features for every frame using 1s of context
    >>> times, features = create_vision_features(decoder, 'x3d_s', window_size=1.0)
    """
    # Route to specialized function based on model type
    if model_name.startswith('x3d'):
        return extract_video_features(images, model_name, window_size=window_size, temporal_subsample=temporal_subsample, verbose=verbose, device=device, **kwargs)
    elif model_name == 'clip':
        from .multimodal import create_multimodal_features
        return create_multimodal_features(model_name, video=images, batch_size=batch_size, device=device)
    else:
        return extract_image_features(images, model_name, batch_size, verbose, device)


def create_motion_energy_features(images, batch_size=None, verbose=True, **kwargs):
    """Extract motion energy features from video using pymoten.

    Uses a pyramid of spatio-temporal Gabor filters to extract motion energy
    features across multiple spatial/temporal frequencies, directions, and positions.

    Parameters
    ----------
    images : torchcodec.decoders.VideoDecoder
        Video decoder from torchcodec with indexing support
    batch_size : int, optional
        Number of frames to process at once. If None, processes entire video.
        Use batching for large/high-resolution videos to manage memory.
    verbose : bool, default=True
        Whether to print progress information
    **kwargs : dict
        Motion energy pyramid parameters. Any parameter not specified will use
        defaults from PYMOTEN_DEFAULT_PARAMS. Common parameters:

        temporal_frequencies : array-like
            Temporal frequencies in Hz (default: [0, 2, 4])
        spatial_frequencies : array-like
            Spatial frequencies in cycles-per-image (default: [0, 2, 4, 8, 16])
        spatial_directions : array-like
            Motion directions in degrees (default: [0, 45, 90, 135, 180, 225, 270, 315])
        filter_temporal_width : int or 'auto'
            Temporal filter window in frames (default: 'auto')
        downsample_factor : int or None
            Factor to downsample frames (e.g., 2 = downsample by 2x) (default: None)

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    motion_features : np.ndarray
        Motion energy features (n_frames, n_filters)

    Examples
    --------
    >>> from torchcodec.decoders import VideoDecoder
    >>> decoder = VideoDecoder("video.mp4", device="cpu")
    >>> times, features = create_motion_energy_features(decoder)

    >>> # Custom frequencies with batching
    >>> times, features = create_motion_energy_features(
    ...     decoder,
    ...     batch_size=500,
    ...     temporal_frequencies=[0, 2, 4, 8],
    ...     downsample_factor=2
    ... )
    """
    import moten
    from moten.pyramids import MotionEnergyPyramid

    # Merge user parameters with defaults
    params = PYMOTEN_DEFAULT_PARAMS.copy()
    params.update(kwargs)

    # Extract processing-specific params (not passed to pyramid)
    downsample_factor = params.pop('downsample_factor')

    # Get video metadata
    video_fps = float(images.metadata.average_fps)
    n_frames = images.metadata.num_frames
    times = np.arange(0, n_frames / video_fps, 1/video_fps)

    if verbose:
        print(f'Extracting motion energy from {n_frames} frames at {video_fps} fps')

    # Get video dimensions (torchcodec returns [C, H, W])
    first_frame = images[0].numpy()
    if first_frame.ndim == 3 and first_frame.shape[0] in [1, 3]:
        # Convert [C, H, W] to [H, W, C]
        first_frame = np.transpose(first_frame, (1, 2, 0))

    vdim, hdim = first_frame.shape[:2]
    if downsample_factor:
        vdim = vdim // downsample_factor
        hdim = hdim // downsample_factor
        if verbose:
            print(f'Downsampling from {first_frame.shape[:2]} to ({vdim}, {hdim}) '
                  f'(factor: {downsample_factor})')

    # Create motion energy pyramid
    pyramid = MotionEnergyPyramid(
        stimulus_vhsize=(vdim, hdim),
        stimulus_fps=int(video_fps),
        **params
    )

    if verbose:
        print(f'Created pyramid: {pyramid}')

    # Process video
    if batch_size is None or batch_size >= n_frames:
        # Process all frames at once
        if verbose:
            print('Loading all frames...')

        # Directly index decoder to get all frames as tensor, then convert to numpy
        frames = images[:n_frames].numpy()

        # Convert from [T, C, H, W] to [T, H, W, C] if needed
        if frames.ndim == 4 and frames.shape[1] in [1, 3]:
            frames = np.transpose(frames, (0, 2, 3, 1))

        # imagearray2luminance expects uint8 array with shape (nimages, vdim, hdim, color)
        target_size = (vdim, hdim) if downsample_factor else None
        luminance = moten.io.imagearray2luminance(frames, size=target_size)

        if verbose:
            print(f'Luminance shape: {luminance.shape}')
            print('Computing motion energy features...')

        motion_features = pyramid.project_stimulus(luminance)

    else:
        # Process in batches with padding for temporal continuity
        if verbose:
            print(f'Processing in batches of {batch_size} frames...')

        # Determine padding size (needs to cover temporal filter width)
        filter_temporal_width = params['filter_temporal_width']
        if filter_temporal_width == 'auto':
            padding = int(np.ceil(2 * video_fps / 3))
        else:
            padding = filter_temporal_width

        motion_features = []
        n_batches = int(np.ceil(n_frames / batch_size))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_frames)

            # Add padding for temporal continuity (except at video edges)
            pad_start = max(0, start_idx - padding) if batch_idx > 0 else start_idx
            pad_end = min(n_frames, end_idx + padding) if batch_idx < n_batches - 1 else end_idx

            if verbose:
                print(f'Batch {batch_idx + 1}/{n_batches}: '
                      f'frames {start_idx}-{end_idx} (padded {pad_start}-{pad_end})')

            # Load frames with padding - directly index decoder to get tensor, then convert to numpy
            frames = images[pad_start:pad_end].numpy()
            # Convert from [T, C, H, W] to [T, H, W, C] if needed
            if frames.ndim == 4 and frames.shape[1] in [1, 3]:
                frames = np.transpose(frames, (0, 2, 3, 1))

            # imagearray2luminance expects uint8 array with shape (nimages, vdim, hdim, color)
            target_size = (vdim, hdim) if downsample_factor else None
            luminance = moten.io.imagearray2luminance(frames, size=target_size)

            # Extract features
            batch_features = pyramid.project_stimulus(luminance)

            # Trim padding from output
            trim_start = start_idx - pad_start
            trim_end = trim_start + (end_idx - start_idx)
            batch_features = batch_features[trim_start:trim_end]

            motion_features.append(batch_features)

        # Concatenate all batches
        motion_features = np.vstack(motion_features)

    if verbose:
        print(f'Motion energy feature shape: {motion_features.shape}')
        print(f'Filters per frame: {motion_features.shape[1]}')

    return times, motion_features


def create_scene_cut_features(video_path, detector='content', threshold=None,
                               min_scene_len=15, dilation_window=0.5, verbose=True, **kwargs):
    """Extract scene cut features from video using PySceneDetect.

    Creates binary features indicating scene transitions/cuts at each frame.
    Useful for encoding models to capture temporal structure of video narratives.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file. PySceneDetect requires direct file access.
    detector : str, default='content'
        Scene detection algorithm to use:
        - 'content': ContentDetector - detects cuts based on content changes (default)
        - 'adaptive': AdaptiveDetector - two-pass detector, better for fast camera movement
        - 'threshold': ThresholdDetector - detects fade in/out events based on intensity
        - 'hash': HashDetector - uses perceptual hashing for scene detection
    threshold : float, optional
        Detection threshold. If None, uses detector-specific defaults:
        - content: 27.0 (higher = fewer detections)
        - adaptive: 3.0 (content_threshold)
        - threshold: 12.0 (lower = fewer detections)
        - hash: 0.395 (higher = fewer detections)
    min_scene_len : int, default=15
        Minimum number of frames for a scene. Helps filter spurious detections.
    dilation_window : float, default=0.5
        Time window in seconds to dilate scene cuts symmetrically. Each detected cut
        will be propagated to frames within +/- dilation_window/2 of the cut time.
        Set to 0 to disable dilation (only mark exact cut frames).
    verbose : bool, default=True
        Whether to print progress information
    **kwargs : dict
        Additional detector-specific parameters:

        For ContentDetector:
            luma_only : bool, default=True
                Only use luma channel for detection (faster)

        For AdaptiveDetector:
            adaptive_threshold : float, default=3.5
                Adaptive threshold for the rolling average
            window_width : int, default=2
                Window size for adaptive comparison

        For ThresholdDetector:
            fade_bias : float, default=0.0
                Bias toward fade-in (-1.0) or fade-out (+1.0)
            add_final_scene : bool, default=True
                Whether to add final scene if video ends mid-scene

        For HashDetector:
            size : int, default=16
                Size of the hash (larger = more sensitive)
            lowpass : int, default=2
                Lowpass filter size

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    scene_features : np.ndarray
        Binary scene cut features (n_frames, 1). Value is 1 at frames where
        a scene cut begins, 0 otherwise.

    Examples
    --------
    >>> # Basic usage with default ContentDetector
    >>> times, features = create_scene_cut_features('video.mp4')

    >>> # Use AdaptiveDetector for videos with fast camera movement
    >>> times, features = create_scene_cut_features(
    ...     'video.mp4', detector='adaptive', threshold=3.0
    ... )

    >>> # Detect fade in/out events
    >>> times, features = create_scene_cut_features(
    ...     'video.mp4', detector='threshold', threshold=12.0
    ... )

    >>> # More sensitive detection (lower threshold = more cuts detected)
    >>> times, features = create_scene_cut_features(
    ...     'video.mp4', detector='content', threshold=20.0
    ... )

    Notes
    -----
    Scene cuts are represented as binary indicators at the first frame of each
    new scene. This is suitable for use in encoding models where you want to
    model the neural response to narrative transitions.

    For fMRI encoding models, you may want to use these features with a Delayer
    to model the hemodynamic response to scene transitions.
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import (
        ContentDetector,
        AdaptiveDetector,
        ThresholdDetector,
        HashDetector
    )

    # Open video to get metadata
    video = open_video(str(video_path))
    video_fps = video.frame_rate
    n_frames = video.duration.get_frames()

    # Create time array
    times = np.arange(0, n_frames / video_fps, 1/video_fps)

    if verbose:
        print(f'Detecting scene cuts in video: {video_path}')
        print(f'  Frames: {n_frames}, FPS: {video_fps:.2f}')
        print(f'  Duration: {n_frames / video_fps:.2f}s')
        print(f'  Detector: {detector}')

    # Create detector with appropriate threshold
    if detector == 'content':
        thresh = threshold if threshold is not None else 27.0
        detector_obj = ContentDetector(
            threshold=thresh,
            min_scene_len=min_scene_len,
            luma_only=kwargs.get('luma_only', True)
        )
    elif detector == 'adaptive':
        thresh = threshold if threshold is not None else 3.0
        detector_obj = AdaptiveDetector(
            adaptive_threshold=thresh,
            min_scene_len=min_scene_len,
            window_width=kwargs.get('window_width', 2),
            min_content_val=kwargs.get('min_content_val', 15.0)
        )
    elif detector == 'threshold':
        thresh = threshold if threshold is not None else 12.0
        detector_obj = ThresholdDetector(
            threshold=thresh,
            min_scene_len=min_scene_len,
            fade_bias=kwargs.get('fade_bias', 0.0),
            add_final_scene=kwargs.get('add_final_scene', True)
        )
    elif detector == 'hash':
        thresh = threshold if threshold is not None else 0.395
        detector_obj = HashDetector(
            threshold=thresh,
            min_scene_len=min_scene_len,
            size=kwargs.get('size', 16),
            lowpass=kwargs.get('lowpass', 2)
        )
    else:
        raise ValueError(f"Unknown detector: {detector}. "
                        f"Must be 'content', 'adaptive', 'threshold', or 'hash'.")

    if verbose:
        print(f'  Threshold: {thresh}')
        print(f'  Min scene length: {min_scene_len} frames')

    # Create scene manager and detect scenes
    scene_manager = SceneManager()
    scene_manager.add_detector(detector_obj)

    if verbose:
        print('Detecting scenes...')

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if verbose:
        print(f'  Found {len(scene_list)} scenes')

    # Create binary feature array (1 at scene cut frames, 0 elsewhere)
    scene_features = np.zeros((len(times), 1), dtype=np.float32)

    # Mark the start frame of each scene (except the first scene at frame 0)
    cut_frames = []
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].frame_num
        # Skip the first scene (frame 0 is not a "cut")
        if start_frame > 0:
            if start_frame < len(scene_features):
                cut_frames.append(start_frame)

    # Apply symmetric dilation to propagate cuts to neighboring frames
    if dilation_window > 0 and len(cut_frames) > 0:
        half_window_frames = int(np.ceil((dilation_window / 2) * video_fps))
        for cut_frame in cut_frames:
            start_frame = max(0, cut_frame - half_window_frames)
            end_frame = min(cut_frame + half_window_frames + 1, len(scene_features))
            scene_features[start_frame:end_frame, 0] = 1.0
    else:
        # No dilation - just mark exact cut frames
        for cut_frame in cut_frames:
            scene_features[cut_frame, 0] = 1.0

    if verbose:
        n_cuts_marked = int(scene_features.sum())
        print(f'  Detected {len(cut_frames)} scene cuts')
        if dilation_window > 0:
            half_window = int(np.ceil((dilation_window / 2) * video_fps))
            print(f'  Dilation window: {dilation_window}s (+/- {half_window} frames)')
            print(f'  Total frames marked: {n_cuts_marked}')
        if len(cut_frames) > 0 and len(cut_frames) <= 20:
            cut_times = [times[f] for f in cut_frames if f < len(times)]
            print(f'  Cut times: {[f"{t:.2f}s" for t in cut_times]}')

    return times, scene_features