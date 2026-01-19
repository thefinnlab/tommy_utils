"""
Dimensionality reduction utilities for neuroimaging feature extraction.

This module provides functions for performing PCA-based dimensionality reduction
with automatic component selection based on variance thresholds.

Example usage:
    >>> from tommy_utils.decomp import fit_and_reduce
    >>> from sklearn.decomposition import IncrementalPCA
    >>> X_reduced, model, info = fit_and_reduce(X, IncrementalPCA, variance_threshold=0.95)
"""

from typing import Optional, Tuple, Dict, List, Type, Any
import numpy as np


def fit_and_reduce(
    X: np.ndarray,
    method: Type = None,
    variance_threshold: float = 0.95,
    n_components: Optional[int] = None,
    method_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Any, Dict]:
    """Fit a decomposition method and reduce data based on variance threshold.

    Performs a two-stage process:
    1. Fit with max components to determine explained variance
    2. Refit with optimal components and transform data

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features).
    method : class, optional
        Sklearn-compatible decomposition class (e.g., PCA, IncrementalPCA).
        Defaults to IncrementalPCA.
    variance_threshold : float, optional
        Target cumulative variance to retain (default: 0.95).
        Ignored if n_components is specified.
    n_components : int, optional
        Specific number of components to use. If provided, variance_threshold
        is ignored.
    method_kwargs : dict, optional
        Additional keyword arguments passed to the decomposition method.
    verbose : bool, optional
        If True, print progress information (default: True).

    Returns
    -------
    X_reduced : np.ndarray
        Transformed data, shape (n_samples, n_components_selected).
    model : object
        Fitted decomposition model.
    info : dict
        Dictionary containing:
        - explained_variance_ratio: variance ratio per component
        - cumulative_variance: cumulative variance
        - n_components: number of components selected
        - variance_explained: actual variance explained
        - n_features_original: original feature count

    Examples
    --------
    >>> from sklearn.decomposition import IncrementalPCA, PCA
    >>> # Using IncrementalPCA with 90% variance
    >>> X_reduced, model, info = fit_and_reduce(X, IncrementalPCA, variance_threshold=0.90)
    >>> # Using standard PCA with specific components
    >>> X_reduced, model, info = fit_and_reduce(X, PCA, n_components=50)
    >>> # Using IncrementalPCA with custom batch size
    >>> X_reduced, model, info = fit_and_reduce(
    ...     X, IncrementalPCA, variance_threshold=0.95,
    ...     method_kwargs={'batch_size': 100}
    ... )
    """
    from sklearn.decomposition import IncrementalPCA

    if method is None:
        method = IncrementalPCA

    method_kwargs = method_kwargs or {}
    n_samples, n_features = X.shape
    n_components_max = min(n_samples, n_features)

    # If n_components specified, skip variance discovery
    if n_components is not None:
        if verbose:
            print(f"Fitting {method.__name__} with {n_components} components...")

        model = method(n_components=n_components, **method_kwargs)
        X_reduced = model.fit_transform(X)

        explained_variance_ratio = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        info = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components': n_components,
            'variance_explained': cumulative_variance[-1],
            'n_features_original': n_features
        }
        return X_reduced, model, info

    # Stage 1: Fit with max components to find variance
    if verbose:
        print(f"Stage 1: Fitting {method.__name__} with {n_components_max} components...")

    model_full = method(n_components=n_components_max, **method_kwargs)
    model_full.fit(X)

    explained_variance_ratio = model_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find components for threshold
    n_components_selected = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    # Handle case where threshold isn't reached
    if cumulative_variance[-1] < variance_threshold:
        n_components_selected = len(cumulative_variance)
        if verbose:
            print(f"  Warning: Max variance ({cumulative_variance[-1]:.4f}) below "
                  f"threshold ({variance_threshold:.4f})")

    if verbose:
        print(f"Stage 2: Refitting with {n_components_selected} components "
              f"({variance_threshold:.0%} variance)...")

    # Stage 2: Refit with selected components
    model = method(n_components=n_components_selected, **method_kwargs)
    X_reduced = model.fit_transform(X)

    variance_explained = cumulative_variance[n_components_selected - 1]

    if verbose:
        print(f"  Variance explained: {variance_explained:.4f}")
        print(f"  Compression: {n_features} -> {n_components_selected} "
              f"({n_features / n_components_selected:.1f}x)")

    info = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components_selected,
        'variance_explained': variance_explained,
        'n_features_original': n_features
    }

    return X_reduced, model, info


def plot_scree(
    info: Dict,
    variance_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 5),
    max_components: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot scree plot showing individual and cumulative explained variance.

    Parameters
    ----------
    info : dict
        Info dictionary returned from fit_and_reduce.
    variance_threshold : float, optional
        Variance threshold to mark. If None, uses 0.95.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    max_components : int, optional
        Maximum components to display. If None, shows all.
    save_path : str, optional
        Path to save the figure.
    show : bool, optional
        If True, display the figure (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    explained_var = info['explained_variance_ratio']
    cumulative_var = info['cumulative_variance']
    n_selected = info['n_components']

    if variance_threshold is None:
        variance_threshold = 0.95

    # Limit display
    if max_components is not None:
        n_display = min(max_components, len(explained_var))
    else:
        n_display = len(explained_var)

    explained_var = explained_var[:n_display]
    cumulative_var = cumulative_var[:n_display]
    components = range(1, n_display + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Individual variance
    axes[0].plot(components, explained_var, 'o-', linewidth=2, markersize=3)
    if n_selected <= n_display:
        axes[0].axvline(x=n_selected, color='r', linestyle='--', linewidth=2,
                       label=f'{n_selected} components')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Explained Variance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative variance
    axes[1].plot(components, cumulative_var, 'o-', linewidth=2, markersize=3, color='green')
    axes[1].axhline(y=variance_threshold, color='r', linestyle='--', linewidth=2,
                   label=f'{variance_threshold:.0%} threshold')
    if n_selected <= n_display:
        axes[1].axvline(x=n_selected, color='r', linestyle='--', linewidth=2,
                       label=f'{n_selected} components')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def get_n_components_for_variance(info: Dict, threshold: float) -> int:
    """Get number of components needed for a variance threshold.

    Parameters
    ----------
    info : dict
        Info dictionary from fit_and_reduce.
    threshold : float
        Variance threshold (0-1).

    Returns
    -------
    int
        Number of components needed.
    """
    return int(np.argmax(info['cumulative_variance'] >= threshold) + 1)
