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
    ]
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

        if flatten:
            arr = torch.flatten(arr, 1, -1)

        layer_tensors.append(arr)
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
    if trim:
        idxs = (i for i in range(trim[0], trim[1]))
    else:
        idxs = (i for i in range(len(decoder)))

    while True:
        sl = list(itertools.islice(idxs, batch_size))
        if not sl:
            break

        frames = [decoder[i] for i in sl]
        yield torch.stack(frames)


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
    # Normalize CLIP features
    return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)


def _extract_cnn_features(model, batch, model_layers, transform):
    """Extract CNN layer activations using torchlens."""
    batch = torch.stack([transform(x) for x in batch])
    model_output = tl.log_forward_pass(model, batch, layers_to_save=model_layers)
    return get_layer_tensors(model_output)


def create_vision_features(images, model_name, batch_size=8, verbose=True):
    """Extract vision features from video frames.

    Parameters
    ----------
    images : video decoder
        Video frame decoder
    model_name : str
        Name of vision model ('alexnet', 'resnet50', 'clip')
    batch_size : int
        Batch size for processing
    verbose : bool
        Whether to print progress

    Returns
    -------
    times : np.ndarray
        Frame times in seconds
    vision_features : np.ndarray or list of np.ndarray
        Vision features (single array for CLIP, list of arrays for layer-wise models)
    """
    # Load model
    tokenizer, model = _get_vision_model(model_name)
    is_clip = model_name == 'clip'

    # Get model layers for CNN models
    model_layers = None if is_clip else VISION_MODELS_DICT[model_name]
    transform = None if is_clip else _get_standard_transform()

    # Compute time points
    video_fps = int(images.metadata.average_fps)
    n_frames = len(images)
    times = np.arange(0, n_frames / video_fps, 1/video_fps)

    # Extract features batch by batch
    vision_features = []
    n_batches = len(images) // batch_size

    for i, batch in enumerate(chunk_video_clips(images, batch_size=batch_size)):
        if verbose:
            print(f'Processing batch {i+1}/{n_batches}')

        if is_clip:
            embeddings = _extract_clip_features(model, tokenizer, batch)
        else:
            embeddings = _extract_cnn_features(model, batch, model_layers, transform)

        vision_features.append(embeddings)

    # Stack features across batches
    if is_clip:
        vision_features = np.vstack(vision_features)
        if verbose:
            print(f'Vision feature space has shape {vision_features.shape}')
    else:
        vision_features = [np.vstack(item) for item in zip(*vision_features)]
        if verbose:
            print(f'Vision feature space has {len(vision_features)} layers')

    return times, vision_features
