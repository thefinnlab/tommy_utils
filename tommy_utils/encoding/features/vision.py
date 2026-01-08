"""Vision feature extraction for encoding models.

This module provides functions for extracting vision features from images/video:
- Pre-trained CNN features (AlexNet, ResNet50)
- CLIP vision features
- Layer-wise feature extraction
"""

import itertools
import numpy as np
import torch
import torchvision
import torchlens as tl
from torchvision import transforms

# Model-specific layer configurations
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


def chunk_video_clips(decoder, batch_size, trim=None):
    """Batch video frames from a decoder.

    Parameters
    ----------
    decoder : video decoder
        Video decoder object with frame indexing
    batch_size : int
        Number of frames per batch
    trim : tuple, optional
        (start, end) frame indices to trim to

    Yields
    ------
    torch.Tensor
        Batched frames
    """
    idxs = (i for i in range(trim[0], trim[1])) if trim else (i for i in range(len(decoder)))
    while True:
        sl = list(itertools.islice(idxs, batch_size))
        if not sl:
            break
        yield torch.stack([decoder[i] for i in sl])


def _get_vision_model(model_name):
    """Load vision model (CNN or CLIP)."""
    if model_name == 'clip':
        from ... import nlp_legacy as nlp
        return nlp.load_multimodal_model(model_name=model_name, modality='vision')
    else:
        model = load_torchvision_model(model_name)
        return None, model


def _get_standard_transform():
    """Get standard ImageNet preprocessing transform."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_clip_features(model, tokenizer, batch):
    """Extract and normalize CLIP image features."""
    inputs = tokenizer(images=batch, return_tensors='pt')
    embeddings = model.get_image_features(**inputs).detach()
    return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)


def _extract_layer_features(model, batch, model_layers, transform):
    """Extract multi-layer activations using torchlens (works with any PyTorch model)."""
    batch = torch.stack([transform(x) for x in batch])
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


def extract_image_features(images, model_name, batch_size=8, verbose=True):
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

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    features : list of np.ndarray
        Multi-layer features, one array per layer
    """
    model = load_torchvision_model(model_name)
    model_layers = VISION_MODELS_DICT[model_name]
    transform = _get_standard_transform()

    video_fps = int(images.metadata.average_fps)
    n_frames = len(images)
    times = np.arange(0, n_frames / video_fps, 1/video_fps)

    vision_features = []
    for i, batch in enumerate(chunk_video_clips(images, batch_size=batch_size)):
        if verbose:
            print(f'Processing batch {i+1}/{n_frames // batch_size}')
        vision_features.append(_extract_layer_features(model, batch, model_layers, transform))

    vision_features = [np.vstack(item) for item in zip(*vision_features)]
    if verbose:
        print(f'Image feature space has {len(vision_features)} layers')

    return times, vision_features


def extract_video_features(images, model_name, clip_duration=None, verbose=True, **kwargs):
    """Extract temporal video features with context (X3D).

    Parameters
    ----------
    images : video decoder
        Video frame decoder with get_frames_in_range() support
    model_name : str
        Video model name ('x3d_s', 'x3d_xs', 'x3d_m')
    clip_duration : float, optional
        Duration of video clip to process in seconds. If None, processes entire video.
    verbose : bool
        Whether to print progress
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
        Frame times in seconds
    features : list of np.ndarray
        Multi-layer video features, one array per layer [n_timepoints, feature_dim]

    Notes
    -----
    This function efficiently extracts features using:
    - Batch frame extraction with get_frames_in_range()
    - UniformTemporalSubsample for temporal downsampling
    - Sliding window over video with model-specific context
    - Temporal reduction to get one feature vector per input frame

    Examples
    --------
    >>> # Extract features with last frame, flattened
    >>> times, features = extract_video_features(decoder, 'x3d_s')

    >>> # Extract features with global average pooling
    >>> times, features = extract_video_features(
    ...     decoder, 'x3d_s', temporal_reduction='mean'
    ... )

    >>> # Extract features preserving spatial dimensions
    >>> times, features = extract_video_features(
    ...     decoder, 'x3d_s', temporal_reduction='last', flatten=False
    ... )
    """
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
    total_frames = metadata.num_frames_from_content

    # Determine frame range to process
    if clip_duration is not None:
        end_frame = min(int(clip_duration * video_fps), total_frames)
    else:
        end_frame = total_frames
    start_frame = 0

    # Calculate times array
    times = np.arange(start_frame, end_frame) / video_fps

    if verbose:
        context_duration = (num_frames - 1) * sampling_rate / video_fps
        print(f'Video model: {model_name}')
        print(f'  Frames per clip: {num_frames}')
        print(f'  Sampling rate: {sampling_rate} ({video_fps/sampling_rate:.1f} fps effective)')
        print(f'  Temporal context: {context_duration:.2f}s')
        print(f'  Processing frames {start_frame}-{end_frame} ({end_frame - start_frame} total)')
        print(f'  Temporal reduction: {temporal_reduction}')

    # Load model
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model.eval()

    # Create transform
    transform = _get_x3d_transform(num_frames, side_size, crop_size)

    # Process video with sliding window (one embedding per frame)
    all_layer_features = [[] for _ in model_layers]

    for current_frame in range(start_frame, end_frame):
        if verbose:
            print(f'Processing frame {current_frame}/{end_frame}')

        # Calculate frame range for this clip's temporal context
        # We need num_frames total, sampled every sampling_rate frames
        required_span = (num_frames - 1) * sampling_rate
        clip_start = max(0, current_frame - required_span)
        clip_end = current_frame + 1

        # Extract frames efficiently in batch
        frames = images.get_frames_in_range(start=clip_start, stop=clip_end)

        # frames.data is [T, C, H, W], convert to [C, T, H, W]
        video_tensor = frames.data.permute(1, 0, 2, 3)

        # Create dictionary format expected by ApplyTransformToKey
        video_data = {"video": video_tensor}

        # Apply transform (includes UniformTemporalSubsample)
        video_data = transform(video_data)
        inputs = video_data["video"].unsqueeze(0)  # [1, C, T, H, W]
        
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

def extract_multimodal_features(images, model_name, batch_size=8, verbose=True):
    """Extract CLIP multimodal features.

    Parameters
    ----------
    images : video decoder
        Video frame decoder
    model_name : str
        Must be 'clip'
    batch_size : int
        Batch size for processing frames
    verbose : bool
        Whether to print progress

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    features : np.ndarray
        CLIP features [n_frames, feature_dim]
    """
    tokenizer, model = _get_vision_model(model_name)

    video_fps = int(images.metadata.average_fps)
    n_frames = len(images)
    times = np.arange(0, n_frames / video_fps, 1/video_fps)

    vision_features = []
    for i, batch in enumerate(chunk_video_clips(images, batch_size=batch_size)):
        if verbose:
            print(f'Processing batch {i+1}/{n_frames // batch_size}')
        vision_features.append(_extract_clip_features(model, tokenizer, batch))

    vision_features = np.vstack(vision_features)
    if verbose:
        print(f'Multimodal feature space has shape {vision_features.shape}')

    return times, vision_features


def create_vision_features(images, model_name, batch_size=8, verbose=True, **kwargs):
    """Extract vision features from video frames.

    Dispatcher function that routes to specialized extraction functions based on model type.

    Parameters
    ----------
    images : video decoder
        Video frame decoder
    model_name : str
        Name of vision model ('alexnet', 'resnet50', 'clip', 'x3d_s')
    batch_size : int
        Batch size for processing
    verbose : bool
        Whether to print progress
    **kwargs : dict
        Model-specific parameters (passed to specialized functions).
        For X3D models:
            clip_duration : float, optional
                Duration of video clip to process in seconds (default: entire video)
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

    >>> # CLIP
    >>> times, features = create_vision_features(decoder, 'clip')

    >>> # X3D with 2-second context @ 24fps
    >>> times, features = create_vision_features(decoder, 'x3d_s', sampling_rate=4)
    """
    # Route to specialized function based on model type
    if model_name.startswith('x3d'):
        # Note: batch_size is not used for X3D models (processes one frame at a time)
        return extract_video_features(images, model_name, verbose=verbose, **kwargs)
    elif model_name == 'clip':
        return extract_multimodal_features(images, model_name, batch_size, verbose)
    else:
        return extract_image_features(images, model_name, batch_size, verbose)


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