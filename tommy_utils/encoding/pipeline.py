"""Pipeline building for encoding models - optimized version."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from himalaya.ridge import ColumnTransformerNoStack, BandedRidgeCV
from himalaya.kernel_ridge import (
    KernelRidgeCV,
    MultipleKernelRidgeCV,
    ColumnKernelizer,
    Kernelizer
)

from .delayer import Delayer
from .solvers.custom_solvers import (
    GroupLevelBandedRidge,
    GroupLevelMultipleKernelRidgeCV,
    solve_group_level_group_ridge_random_search,
    solve_group_level_multiple_kernel_ridge_random_search
)
from himalaya.backend import get_backend

# Register custom solvers with Himalaya
BandedRidgeCV.ALL_SOLVERS['group_level_random_search'] = (
    solve_group_level_group_ridge_random_search
)
MultipleKernelRidgeCV.ALL_SOLVERS['group_level_random_search'] = (
    solve_group_level_multiple_kernel_ridge_random_search
)


def _create_base_solver_params(n_iter, alphas, n_targets_batch,
                               n_alphas_batch, n_targets_batch_refit):
    """Create base solver parameters dict."""
    return {
        'n_iter': n_iter,
        'alphas': alphas,
        'n_targets_batch': n_targets_batch,
        'n_alphas_batch': n_alphas_batch,
        'n_targets_batch_refit': n_targets_batch_refit
    }


def _validate_group_level_shapes(Y):
    """Validate that all Y arrays have the same shape for group-level modeling."""
    if not all([y.shape == Y[0].shape for y in Y]):
        raise ValueError(
            "To use group level random search, all "
            "groups need to have same number of samples."
        )
    return Y[0].shape[0]


def _create_ridge_model(model_class, solver, solver_params, cv,
                       Y_in_cpu=False, force_cpu=False, **kwargs):
    """Create a ridge model with common parameters."""
    return model_class(
        solver=solver,
        solver_params=solver_params,
        cv=cv,
        Y_in_cpu=Y_in_cpu,
        force_cpu=force_cpu,
        **kwargs
    )


def create_banded_model(model, delays, feature_space_infos, kernel=None,
                       n_jobs=None, force_cpu=False):
    """Create a banded model with preprocessing pipeline.

    Parameters
    ----------
    model : sklearn estimator
        Ridge model (BandedRidgeCV, MultipleKernelRidgeCV, etc.)
    delays : list of int
        HRF delays to model
    feature_space_infos : list of tuple
        Names and slices for each feature space
    kernel : str, optional
        Kernel function for kernelized models
    n_jobs : int, optional
        Number of parallel jobs
    force_cpu : bool
        Force CPU computation

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Complete model pipeline with preprocessing
    """
    # Standard preprocessing: demean (but keep std as it contains information)
    scaler = StandardScaler(with_mean=True, with_std=False)
    delayer = Delayer(delays=delays)

    # Build preprocessing pipeline
    preprocess_steps = [scaler, delayer]
    if kernel:
        preprocess_steps.append(Kernelizer(kernel=kernel))
    preprocess_pipeline = make_pipeline(*preprocess_steps)

    # Create feature tuples for each feature space
    feature_tuples = [
        (name, preprocess_pipeline, slice_)
        for name, slice_ in feature_space_infos
    ]

    # Select appropriate column transformer
    if kernel:
        column_transformer = ColumnKernelizer(
            feature_tuples,
            n_jobs=n_jobs,
            force_cpu=force_cpu
        )
    else:
        column_transformer = ColumnTransformerNoStack(
            feature_tuples,
            n_jobs=n_jobs
        )

    return make_pipeline(column_transformer, model)


def build_encoding_pipeline(X, Y, inner_cv, feature_space_infos=None,
                            delays=[1,2,3,4], n_iter=20, n_targets_batch=200,
                            n_alphas_batch=5, n_targets_batch_refit=200,
                            Y_in_cpu=False, force_cpu=False,
                            solver="random_search",
                            alphas=np.logspace(1, 20, 20),
                            n_jobs=None, force_banded_ridge=False):
    """Build an encoding model pipeline with ridge regression.

    This function selects the appropriate model type based on the data dimensions
    and feature space configuration:
    - Single feature space: KernelRidgeCV
    - Multiple feature spaces + n_samples > n_features: BandedRidgeCV
    - Multiple feature spaces + n_samples < n_features: MultipleKernelRidgeCV

    Parameters
    ----------
    X : list of np.ndarray
        Feature arrays
    Y : list of np.ndarray
        Target arrays
    inner_cv : int or cross-validation generator
        Inner cross-validation strategy
    feature_space_infos : list of tuple, optional
        Feature space names and slices for banded ridge
    delays : list of int
        HRF delays to model
    n_iter : int
        Number of random search iterations
    n_targets_batch : int
        Batch size for targets during CV
    n_alphas_batch : int
        Batch size for alphas
    n_targets_batch_refit : int
        Batch size for targets during refit
    Y_in_cpu : bool
        Keep Y in CPU memory
    force_cpu : bool
        Force CPU computation
    solver : str
        Solver name ('random_search', 'group_level_random_search', 'hyper_gradient')
    alphas : np.ndarray
        Alpha values to search
    n_jobs : int, optional
        Number of parallel jobs
    force_banded_ridge : bool
        Force banded ridge even when n_samples < n_features

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Complete encoding pipeline
    """
    # Validate input shapes
    if solver == 'group_level_random_search':
        assert all([X[0].shape[0] == y.shape[0] for y in Y])
    else:
        assert len(X) == len(Y)

    n_samples = np.concatenate(X).shape[0]
    n_features = np.concatenate(X).shape[1]

    # Single feature space: use standard kernel ridge
    if not feature_space_infos:
        solver_params = _create_base_solver_params(
            n_iter, alphas, n_targets_batch,
            n_alphas_batch, n_targets_batch_refit
        )

        ridge = KernelRidgeCV(
            kernel="linear",
            alphas=alphas,
            cv=inner_cv,
            Y_in_cpu=Y_in_cpu,
            force_cpu=force_cpu
        )

        scaler = StandardScaler(with_mean=True, with_std=False)
        delayer = Delayer(delays=delays)
        return make_pipeline(scaler, delayer, ridge)

    # Multiple feature spaces: select appropriate model
    use_banded = n_samples > n_features or force_banded_ridge

    print(f'Using {"banded ridge" if use_banded else "multiple kernel ridge"}')

    # Prepare base solver parameters
    base_solver_params = _create_base_solver_params(
        n_iter, alphas, n_targets_batch,
        n_alphas_batch, n_targets_batch_refit
    )

    if use_banded:
        # Banded Ridge models
        if solver == 'group_level_random_search':
            n_samples_group = _validate_group_level_shapes(Y)
            solver_params = {**base_solver_params, 'n_samples_group': n_samples_group}

            model = _create_ridge_model(
                GroupLevelBandedRidge,
                solver, solver_params, inner_cv,
                Y_in_cpu, force_cpu,
                groups="input"
            )
        elif solver == 'random_search':
            model = _create_ridge_model(
                BandedRidgeCV,
                solver, base_solver_params, inner_cv,
                Y_in_cpu, force_cpu,
                groups="input"
            )
        else:
            raise ValueError(f"Unsupported solver for banded ridge: {solver}")

        return create_banded_model(
            model, delays, feature_space_infos, n_jobs=n_jobs
        )

    else:
        # Multiple Kernel Ridge models
        if solver == 'group_level_random_search':
            n_samples_group = _validate_group_level_shapes(Y)
            solver_params = {
                **base_solver_params,
                'n_samples_group': n_samples_group,
                'Ks_in_cpu': force_cpu
            }

            model = _create_ridge_model(
                GroupLevelMultipleKernelRidgeCV,
                solver, solver_params, inner_cv,
                Y_in_cpu=Y_in_cpu,
                kernels="precomputed"
            )

        elif solver == 'random_search':
            solver_params = {**base_solver_params, 'Ks_in_cpu': force_cpu}

            model = _create_ridge_model(
                MultipleKernelRidgeCV,
                solver, solver_params, inner_cv,
                Y_in_cpu=Y_in_cpu,
                kernels="precomputed"
            )

        elif solver == 'hyper_gradient':
            solver_params = {
                'max_iter': n_iter,
                'n_targets_batch': n_targets_batch,
                'tol': 1e-3,
                'initial_deltas': "ridgecv",
                'max_iter_inner_hyper': 1,
                'hyper_gradient_method': "direct"
            }

            model = _create_ridge_model(
                MultipleKernelRidgeCV,
                solver, solver_params, inner_cv,
                Y_in_cpu=Y_in_cpu,
                kernels="precomputed"
            )
        else:
            raise ValueError(f"Unsupported solver for kernel ridge: {solver}")

        return create_banded_model(
            model, delays, feature_space_infos,
            kernel="linear", n_jobs=n_jobs, force_cpu=force_cpu
        )


def refine_encoding_model(fitted_pipeline, X_train, Y_train,
                          max_iter=10, max_iter_inner_hyper=10,
                          hyper_gradient_method="direct",
                          n_targets_batch=200, tol=1e-3,
                          top_percentile=None, Y_in_cpu=False):
    """Refine a MultipleKernelRidgeCV model using gradient descent.

    Takes a fitted pipeline (trained with random_search) and refines the
    hyperparameters using gradient descent. Reuses the preprocessing components
    from the original pipeline.

    Based on: https://gallantlab.org/himalaya/_auto_examples/multiple_kernel_ridge/plot_mkr_5_refine_results.html

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with MultipleKernelRidgeCV as the last step
    X_train : np.ndarray
        Training features (same as used to fit the original model)
    Y_train : np.ndarray
        Training targets (same as used to fit the original model)
    max_iter : int, default=10
        Maximum gradient descent iterations
    max_iter_inner_hyper : int, default=10
        Maximum inner iterations for hyper-gradient solver
    hyper_gradient_method : str, default="direct"
        Method for computing hyperparameter gradients ("direct" or "implicit")
    n_targets_batch : int, default=200
        Batch size for targets during optimization
    tol : float, default=1e-3
        Convergence tolerance
    top_percentile : float, optional
        If provided (e.g., 60), only refine top N% of targets by CV score.
        If None, refine all targets.
    Y_in_cpu : bool, default=False
        Keep Y in CPU memory

    Returns
    -------
    refined_pipeline : sklearn.pipeline.Pipeline
        Pipeline with refined hyperparameters
    target_mask : np.ndarray or None
        Boolean mask of refined targets (if top_percentile used), else None

    Examples
    --------
    >>> # Fit with random search
    >>> pipeline = build_encoding_pipeline(..., solver="random_search")
    >>> pipeline.fit(X_train, Y_train)
    >>>
    >>> # Refine with gradient descent
    >>> refined_pipeline, mask = refine_encoding_model(
    ...     pipeline, X_train, Y_train, max_iter=10, top_percentile=60
    ... )
    >>>
    >>> # Predict
    >>> Y_pred = pipeline.predict(X_test)
    >>> if mask is not None:
    ...     Y_pred[:, mask] = refined_pipeline.predict(X_test)
    """
    backend = get_backend()

    # Extract fitted model (last step in pipeline)
    fitted_model = fitted_pipeline[-1]

    # Check if model is MultipleKernelRidgeCV
    if not isinstance(fitted_model, MultipleKernelRidgeCV):
        raise ValueError(
            "This function only works with MultipleKernelRidgeCV models. "
            f"Got {type(fitted_model).__name__} instead."
        )

    if not hasattr(fitted_model, 'deltas_'):
        raise ValueError("Model must be fitted and have 'deltas_' attribute.")

    # Determine which targets to refine
    target_mask = None
    Y_train_subset = Y_train
    initial_deltas = fitted_model.deltas_

    if top_percentile is not None:
        # Select top percentile of targets by CV score
        best_cv_scores = backend.to_numpy(fitted_model.cv_scores_.max(0))
        threshold = np.percentile(best_cv_scores, 100 - top_percentile)
        target_mask = best_cv_scores > threshold

        print(f"Refining {target_mask.sum()} / {len(target_mask)} targets "
              f"(top {top_percentile}%)")

        Y_train_subset = Y_train[:, target_mask]
        initial_deltas = fitted_model.deltas_[:, target_mask]
    else:
        print(f"Refining all {Y_train.shape[1]} targets")

    # Create gradient descent solver parameters
    solver_params = {
        'max_iter': max_iter,
        'max_iter_inner_hyper': max_iter_inner_hyper,
        'hyper_gradient_method': hyper_gradient_method,
        'n_targets_batch': n_targets_batch,
        'tol': tol,
        'initial_deltas': initial_deltas  # Warm start from random search
    }

    # Create refined model with hyper_gradient solver
    refined_model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver='hyper_gradient',
        solver_params=solver_params,
        cv=fitted_model.cv,
        Y_in_cpu=Y_in_cpu
    )

    # Reuse preprocessing from original pipeline (all steps except the last)
    preprocessing_steps = fitted_pipeline.steps[:-1]
    refined_pipeline = make_pipeline(*[step[1] for step in preprocessing_steps],
                                     refined_model)

    # Fit refined pipeline
    print("Fitting refined model with gradient descent...")
    refined_pipeline.fit(X_train, Y_train_subset)
    print("Refinement complete!")

    return refined_pipeline, target_mask
