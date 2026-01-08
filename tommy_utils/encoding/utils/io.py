"""Input/output utilities for saving and loading models."""

from himalaya.backend import get_backend
from .evaluation import BANDED_RIDGE_MODELS, KERNEL_RIDGE_MODELS


def save_model_parameters(pipeline):
    """Save model parameters to dictionary for serialization.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted encoding pipeline

    Returns
    -------
    d : dict
        Dictionary containing model info and hyperparameters
    """
    backend = get_backend()
    d = {}

    d['info'] = {
        'module': pipeline[-1].__class__.__module__,
        'name': pipeline[-1].__class__.__name__,
    }

    if d['info']['name'] in BANDED_RIDGE_MODELS:
        d['hyperparameters'] = {
            'deltas_': backend.to_cpu(pipeline[-1].__dict__['deltas_']),
            'coef_': backend.to_cpu(pipeline[-1].__dict__['coef_'])
        }
    elif d['info']['name'] in KERNEL_RIDGE_MODELS:
        d['hyperparameters'] = {
            'deltas_': backend.to_cpu(pipeline[-1].__dict__['deltas_']),
            'dual_coef': backend.to_cpu(pipeline[-1].__dict__['dual_coef_'])
        }
    else:
        raise ValueError(
            f'Model must be a form of banded ridge or kernel ridge model'
        )

    return d


def load_model_from_parameters(d, args={}):
    """Load a model from saved parameters.

    Parameters
    ----------
    d : dict
        Dictionary containing model info and hyperparameters
    args : dict
        Arguments to pass to model constructor

    Returns
    -------
    model
        Reconstructed model with loaded parameters
    """
    # Make sure we use the backend to cast to type
    backend = get_backend()

    module = __import__(d['info']['module'], fromlist=[d['info']['name']])
    base_ = getattr(module, d['info']['name'])(**args)

    for k, v in d['hyperparameters'].items():
        base_.__dict__[k] = backend.to_cpu(v)

    return base_
