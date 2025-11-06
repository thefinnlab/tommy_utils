"""Model evaluation utilities for encoding models."""

import numpy as np
import himalaya
import himalaya.scoring
from himalaya.backend import get_backend


# Model type classifications for extracting parameters
BANDED_RIDGE_MODELS = [
    'GroupRidgeCV',
    'BandedRidgeCV',
    'GroupLevelBandedRidgeCV',
]

KERNEL_RIDGE_MODELS = [
    'KernelRidgeCV',
    'MultipleKernelRidgeCV',
    'GroupLevelMultipleKernelRidgeCV'
]


def get_all_banded_metrics(pipeline, X_test, Y_test, use_split=False):
    """Compute comprehensive metrics for encoding model.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted encoding pipeline
    X_test : np.ndarray
        Test features
    Y_test : np.ndarray
        Test targets
    use_split : bool
        Whether to compute split metrics

    Returns
    -------
    results : dict
        Dictionary containing predictions, correlations, R2, and residuals
    """
    backend = get_backend()

    # Get reference array for type casting
    if pipeline[-1].__class__.__name__ in BANDED_RIDGE_MODELS:
        ref_arr = pipeline[-1].__dict__['coef_']
    elif pipeline[-1].__class__.__name__ in KERNEL_RIDGE_MODELS:
        ref_arr = pipeline[-1].__dict__['dual_coef_']
    else:
        raise ValueError(
            f'Model must be a form of banded ridge or kernel ridge model'
        )

    X_test = backend.asarray_like(X_test, ref_arr)
    Y_test = backend.asarray_like(Y_test, ref_arr)

    results = {}

    metrics = {
        'correlation': getattr(himalaya.scoring, 'correlation_score'),
        'correlation-split': getattr(himalaya.scoring, 'correlation_score_split'),
        'r2': getattr(himalaya.scoring, 'r2_score'),
        'r2-split': getattr(himalaya.scoring, 'r2_score_split')
    }

    # Predict and make as same type of array as Y_test
    Y_pred = pipeline.predict(X_test)
    Y_pred = backend.asarray_like(Y_pred, Y_test)
    results['prediction'] = Y_pred

    if use_split:
        Y_pred_split = pipeline.predict(X_test, split=True)
        Y_pred_split = backend.asarray_like(Y_pred_split, Y_test)
        results['prediction-split'] = Y_pred_split

    for metric, fx in metrics.items():
        if 'split' in metric:
            if use_split:
                score = fx(Y_test, Y_pred)
            else:
                continue
        else:
            score = fx(Y_test, Y_pred)

        results[metric] = score

    # Calculate residuals
    results['residuals'] = (Y_test - results['prediction'])

    if use_split:
        results['residuals-split'] = (Y_test - results['prediction-split'])

    # Move to CPU and convert to numpy
    results = {k: np.asarray(backend.to_cpu(v)) for k, v in results.items()}

    return results
