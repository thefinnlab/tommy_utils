"""
Dimensionality reduction utilities for neuroimaging feature extraction.

This module provides functions for performing PCA-based dimensionality reduction
with automatic component selection based on variance thresholds.

Key classes:
    - ProgressPCA: IncrementalPCA wrapper with tqdm progress tracking

Key functions:
    - fit_and_reduce: Fit decomposition and reduce data based on variance threshold
    - save_decomposition: Save fitted model and info to disk
    - load_decomposition: Load saved model and info from disk
    - fit_reduce_or_load: Convenience function with automatic caching

Example usage:
    >>> from tommy_utils.decomp import fit_and_reduce, save_decomposition, load_decomposition
    >>> from sklearn.decomposition import IncrementalPCA

    >>> # Fit and reduce
    >>> X_reduced, model, info = fit_and_reduce(X, IncrementalPCA, variance_threshold=0.95)

    >>> # Save for later use
    >>> save_decomposition(model, info, 'cache/my_pca')

    >>> # Load and apply to new data
    >>> model, info = load_decomposition('cache/my_pca')
    >>> X_new_reduced = model.transform(X_new)

    >>> # Or use the convenience function with automatic caching
    >>> from tommy_utils.decomp import fit_reduce_or_load
    >>> X_reduced, model, info = fit_reduce_or_load(X, 'cache/my_pca', variance_threshold=0.95)
"""

from typing import Optional, Tuple, Dict, List, Type, Any, Union, Iterator
from pathlib import Path
import numpy as np


class ProgressPCA:
    """Wrapper for IncrementalPCA that displays progress during fitting.

    This class wraps sklearn's IncrementalPCA to show a tqdm progress bar
    during partial_fit operations. It can be used either with manual batch
    iteration or by passing the full data array.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If None, keeps min(n_samples, n_features).
    batch_size : int, optional
        Number of samples per batch for partial_fit (default: 256).
    **kwargs
        Additional arguments passed to IncrementalPCA.

    Attributes
    ----------
    ipca : IncrementalPCA
        The underlying IncrementalPCA instance.
    n_samples_seen_ : int
        Number of samples processed so far.

    Examples
    --------
    >>> # Option 1: Fit on full array with automatic batching
    >>> pca = ProgressPCA(n_components=50, batch_size=256)
    >>> pca.fit(X)
    >>> X_reduced = pca.transform(X)

    >>> # Option 2: Manual batch iteration
    >>> pca = ProgressPCA(n_components=50)
    >>> for batch in data_generator:
    ...     pca.partial_fit(batch)
    >>> X_reduced = pca.transform(X)

    >>> # Option 3: Fit with progress on pre-defined batches
    >>> pca = ProgressPCA(n_components=50)
    >>> pca.fit_batches(batches, n_batches=len(batches))
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        batch_size: int = 256,
        **kwargs
    ):
        from sklearn.decomposition import IncrementalPCA

        self.n_components = n_components
        self.batch_size = batch_size
        # Don't pass batch_size to IncrementalPCA - it's only for internal use
        # and conflicts with partial_fit usage
        self.ipca = IncrementalPCA(n_components=n_components, **kwargs)
        self.n_samples_seen_ = 0

    def partial_fit(self, X: np.ndarray) -> "ProgressPCA":
        """Incrementally fit on a batch of samples.

        Parameters
        ----------
        X : np.ndarray
            Batch of samples, shape (n_samples, n_features).

        Returns
        -------
        self
        """
        self.ipca.partial_fit(X)
        self.n_samples_seen_ += X.shape[0]
        return self

    def fit(
        self,
        X: np.ndarray,
        desc: str = "Fitting PCA",
        disable_progress: bool = False
    ) -> "ProgressPCA":
        """Fit on full data array with progress tracking.

        Automatically batches the data and shows a tqdm progress bar.
        Follows sklearn's IncrementalPCA batching pattern.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        desc : str, optional
            Description for the progress bar (default: "Fitting PCA").
        disable_progress : bool, optional
            If True, disable the progress bar (default: False).

        Returns
        -------
        self
        """
        from tqdm.auto import tqdm

        n_samples, n_features = X.shape

        # Follow sklearn's IncrementalPCA pattern:
        # Ensure batch_size is large enough for n_components
        effective_batch_size = self.batch_size
        if self.n_components is not None:
            effective_batch_size = max(effective_batch_size, self.n_components)

        # Also ensure it's at least as large as n_features for efficiency
        effective_batch_size = max(effective_batch_size, n_features)

        # But don't exceed n_samples
        effective_batch_size = min(effective_batch_size, n_samples)

        n_batches = int(np.ceil(n_samples / effective_batch_size))

        with tqdm(total=n_batches, desc=desc, disable=disable_progress) as pbar:
            for i in range(0, n_samples, effective_batch_size):
                batch = X[i:i + effective_batch_size]
                self.partial_fit(batch)
                pbar.update(1)

        return self

    def fit_batches(
        self,
        batches: Iterator[np.ndarray],
        n_batches: Optional[int] = None,
        desc: str = "Fitting PCA",
        disable_progress: bool = False
    ) -> "ProgressPCA":
        """Fit on an iterator of batches with progress tracking.

        Parameters
        ----------
        batches : Iterator[np.ndarray]
            Iterator yielding batches of samples.
        n_batches : int, optional
            Total number of batches (for progress bar). If None, progress
            bar won't show percentage.
        desc : str, optional
            Description for the progress bar (default: "Fitting PCA").
        disable_progress : bool, optional
            If True, disable the progress bar (default: False).

        Returns
        -------
        self
        """
        from tqdm.auto import tqdm

        for batch in tqdm(batches, total=n_batches, desc=desc, disable=disable_progress):
            self.partial_fit(batch)

        return self

    def fit_transform(
        self,
        X: np.ndarray,
        desc: str = "Fitting PCA",
        disable_progress: bool = False
    ) -> np.ndarray:
        """Fit and transform data with progress tracking.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        desc : str, optional
            Description for the progress bar.
        disable_progress : bool, optional
            If True, disable the progress bar.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data, shape (n_samples, n_components).
        """
        self.fit(X, desc=desc, disable_progress=disable_progress)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Data to transform, shape (n_samples, n_features).

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data, shape (n_samples, n_components).
        """
        return self.ipca.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back to original space.

        Parameters
        ----------
        X : np.ndarray
            Transformed data, shape (n_samples, n_components).

        Returns
        -------
        X_original : np.ndarray
            Data in original space, shape (n_samples, n_features).
        """
        return self.ipca.inverse_transform(X)

    @property
    def components_(self) -> np.ndarray:
        """Principal components."""
        return self.ipca.components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Explained variance ratio for each component."""
        return self.ipca.explained_variance_ratio_

    @property
    def explained_variance_(self) -> np.ndarray:
        """Explained variance for each component."""
        return self.ipca.explained_variance_

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature mean estimated from training data."""
        return self.ipca.mean_

    @property
    def n_components_(self) -> int:
        """Number of components."""
        return self.ipca.n_components_


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

    # Check if we should use ProgressPCA for progress tracking
    use_progress = method is IncrementalPCA or method.__name__ == 'IncrementalPCA'

    # If n_components specified, skip variance discovery
    if n_components is not None:
        if verbose:
            print(f"Fitting {method.__name__} with {n_components} components...")

        if use_progress:
            pca = ProgressPCA(n_components=n_components, **method_kwargs)
            X_reduced = pca.fit_transform(X, disable_progress=not verbose)
            model = pca.ipca
        else:
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

    if use_progress:
        pca_full = ProgressPCA(n_components=n_components_max, **method_kwargs)
        pca_full.fit(X, desc="Stage 1: Finding variance", disable_progress=not verbose)
        model_full = pca_full.ipca
    else:
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
    if use_progress:
        pca = ProgressPCA(n_components=n_components_selected, **method_kwargs)
        X_reduced = pca.fit_transform(X, desc="Stage 2: Final fit", disable_progress=not verbose)
        model = pca.ipca
    else:
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


def save_decomposition(
    model: Any,
    info: Dict,
    save_path: Union[str, Path],
    compress: int = 3
) -> None:
    """Save a fitted decomposition model and its info to disk.

    Uses joblib for efficient serialization of sklearn models.

    Parameters
    ----------
    model : object
        Fitted decomposition model (e.g., IncrementalPCA, PCA).
    info : dict
        Info dictionary returned from fit_and_reduce.
    save_path : str or Path
        Path to save the model. Will create two files:
        - {save_path}.joblib: the model
        - {save_path}_info.npz: the info dictionary
    compress : int, optional
        Compression level (0-9). Default is 3, which provides good
        balance between speed and file size.

    Examples
    --------
    >>> X_reduced, model, info = fit_and_reduce(X, IncrementalPCA, variance_threshold=0.95)
    >>> save_decomposition(model, info, 'models/my_pca')
    >>> # Creates: models/my_pca.joblib and models/my_pca_info.npz
    """
    import joblib

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model using joblib
    model_path = save_path.with_suffix('.joblib')
    joblib.dump(model, model_path, compress=compress)

    # Save the info dictionary as npz
    info_path = str(save_path) + '_info.npz'
    np.savez(
        info_path,
        explained_variance_ratio=info['explained_variance_ratio'],
        cumulative_variance=info['cumulative_variance'],
        n_components=info['n_components'],
        variance_explained=info['variance_explained'],
        n_features_original=info['n_features_original']
    )


def load_decomposition(
    load_path: Union[str, Path]
) -> Tuple[Any, Dict]:
    """Load a saved decomposition model and its info from disk.

    Parameters
    ----------
    load_path : str or Path
        Path to the saved model (without extension). Expects:
        - {load_path}.joblib: the model
        - {load_path}_info.npz: the info dictionary

    Returns
    -------
    model : object
        The loaded decomposition model.
    info : dict
        The info dictionary with keys:
        - explained_variance_ratio: variance ratio per component
        - cumulative_variance: cumulative variance
        - n_components: number of components selected
        - variance_explained: actual variance explained
        - n_features_original: original feature count

    Examples
    --------
    >>> model, info = load_decomposition('models/my_pca')
    >>> X_new_reduced = model.transform(X_new)
    """
    import joblib

    load_path = Path(load_path)

    # Load the model
    model_path = load_path.with_suffix('.joblib')
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    # Load the info dictionary
    info_path = str(load_path) + '_info.npz'
    if not Path(info_path).exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")

    with np.load(info_path) as data:
        info = {
            'explained_variance_ratio': data['explained_variance_ratio'],
            'cumulative_variance': data['cumulative_variance'],
            'n_components': int(data['n_components']),
            'variance_explained': float(data['variance_explained']),
            'n_features_original': int(data['n_features_original'])
        }

    return model, info


def fit_reduce_or_load(
    X: np.ndarray,
    save_path: Union[str, Path],
    method: Type = None,
    variance_threshold: float = 0.95,
    n_components: Optional[int] = None,
    method_kwargs: Optional[Dict[str, Any]] = None,
    force_refit: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, Any, Dict]:
    """Fit and reduce data, or load from cache if available.

    This is a convenience function that combines fit_and_reduce with
    save/load functionality. If a saved model exists at save_path and
    force_refit is False, it loads the model and transforms the data.
    Otherwise, it fits a new model, saves it, and returns the results.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features).
    save_path : str or Path
        Path for saving/loading the model.
    method : class, optional
        Sklearn-compatible decomposition class. Defaults to IncrementalPCA.
    variance_threshold : float, optional
        Target cumulative variance to retain (default: 0.95).
    n_components : int, optional
        Specific number of components to use.
    method_kwargs : dict, optional
        Additional keyword arguments passed to the decomposition method.
    force_refit : bool, optional
        If True, always refit even if saved model exists (default: False).
    verbose : bool, optional
        If True, print progress information (default: True).

    Returns
    -------
    X_reduced : np.ndarray
        Transformed data, shape (n_samples, n_components_selected).
    model : object
        Fitted decomposition model.
    info : dict
        Dictionary containing model metadata.

    Examples
    --------
    >>> # First call: fits and saves
    >>> X_reduced, model, info = fit_reduce_or_load(
    ...     X, 'cache/video_pca', variance_threshold=0.95
    ... )
    >>> # Second call: loads from cache
    >>> X_reduced, model, info = fit_reduce_or_load(
    ...     X, 'cache/video_pca', variance_threshold=0.95
    ... )
    """
    save_path = Path(save_path)
    model_path = save_path.with_suffix('.joblib')

    # Check if cached model exists
    if model_path.exists() and not force_refit:
        if verbose:
            print(f"Loading cached model from {model_path}...")
        model, info = load_decomposition(save_path)
        X_reduced = model.transform(X)
        if verbose:
            print(f"  Loaded model with {info['n_components']} components "
                  f"({info['variance_explained']:.4f} variance explained)")
        return X_reduced, model, info

    # Fit new model
    if verbose and model_path.exists():
        print(f"Force refitting (force_refit=True)...")

    X_reduced, model, info = fit_and_reduce(
        X,
        method=method,
        variance_threshold=variance_threshold,
        n_components=n_components,
        method_kwargs=method_kwargs,
        verbose=verbose
    )

    # Save the model
    if verbose:
        print(f"Saving model to {save_path}...")
    save_decomposition(model, info, save_path)

    return X_reduced, model, info
