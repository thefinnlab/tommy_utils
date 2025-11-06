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
