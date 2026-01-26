"""Test script to verify refine_encoding_model works with Himalaya example."""

import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.utils import generate_multikernel_dataset
from sklearn.pipeline import make_pipeline

# Set backend
backend = set_backend("numpy", on_error="warn")

print("Generating synthetic dataset...")
(X_train, X_test, Y_train, Y_test, kernel_weights,
 n_features_list) = generate_multikernel_dataset(
    n_kernels=4, n_targets=50,
    n_samples_train=600,
    n_samples_test=300,
    random_state=42)

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")

# Prepare pipeline with kernelizers
start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
slices = [slice(start, end)
          for start, end in zip(start_and_end[:-1], start_and_end[1:])]

kernelizers = [("space %d" % ii, Kernelizer(), slice_)
               for ii, slice_ in enumerate(slices)]
column_kernelizer = ColumnKernelizer(kernelizers)

# Stage 1: Random search
print("\n=== Stage 1: Random Search ===")
solver_params_1 = dict(n_iter=5, alphas=np.logspace(-10, 10, 41))
model_1 = MultipleKernelRidgeCV(kernels="precomputed",
                                solver="random_search",
                                solver_params=solver_params_1,
                                random_state=42)
pipe_1 = make_pipeline(column_kernelizer, model_1)
pipe_1.fit(X_train, Y_train)
print("Random search complete!")

# Check initial deltas
print(f"\ncv_scores_ shape: {pipe_1[-1].cv_scores_.shape}")
print(f"deltas_ shape: {pipe_1[-1].deltas_.shape}")
print(f"deltas_ range: [{pipe_1[-1].deltas_.min():.2f}, {pipe_1[-1].deltas_.max():.2f}]")

# Stage 2: Test with direct method (what user is using)
print("\n=== Stage 2: Gradient Descent Refinement (direct method) ===")

# Select top performing targets
top = 75
best_cv_scores = backend.to_numpy(pipe_1[-1].cv_scores_.max(0))
mask = best_cv_scores > np.percentile(best_cv_scores, 100 - top)
print(f"Refining {mask.sum()} / {len(mask)} targets (top {top}%)")

# Get initial deltas
initial_deltas = pipe_1[-1].deltas_[:, mask]
print(f"Initial deltas shape: {initial_deltas.shape}")
print(f"Initial deltas range: [{initial_deltas.min():.2f}, {initial_deltas.max():.2f}]")
print(f"Any NaN: {np.any(np.isnan(initial_deltas))}")
print(f"Any Inf: {np.any(np.isinf(initial_deltas))}")

# Clip deltas like in refine_encoding_model
initial_deltas_clipped = np.clip(initial_deltas, -20, 20)
print(f"Clipped deltas range: [{initial_deltas_clipped.min():.2f}, {initial_deltas_clipped.max():.2f}]")

# Test with direct method
print("\n--- Testing direct method ---")
solver_params_2 = dict(
    max_iter=10,
    hyper_gradient_method="direct",
    max_iter_inner_hyper=10,
    initial_deltas=initial_deltas_clipped
)
model_2 = MultipleKernelRidgeCV(kernels="precomputed",
                                solver="hyper_gradient",
                                solver_params=solver_params_2)
pipe_2 = make_pipeline(column_kernelizer, model_2)

try:
    pipe_2.fit(X_train, Y_train[:, mask])
    print("direct: SUCCESS!")
    test_score = pipe_2.score(X_test, Y_test[:, mask])
    print(f"Test R2 mean: {np.mean(backend.to_numpy(test_score)):.4f}")
except AssertionError as e:
    print(f"direct: FAILED with AssertionError - {e}")
except Exception as e:
    print(f"direct: FAILED - {type(e).__name__}: {e}")

# Test with conjugate_gradient method for comparison
print("\n--- Testing conjugate_gradient method ---")
solver_params_3 = dict(
    max_iter=10,
    hyper_gradient_method="conjugate_gradient",
    max_iter_inner_hyper=10,
    initial_deltas=initial_deltas_clipped
)
model_3 = MultipleKernelRidgeCV(kernels="precomputed",
                                solver="hyper_gradient",
                                solver_params=solver_params_3)
pipe_3 = make_pipeline(column_kernelizer, model_3)

try:
    pipe_3.fit(X_train, Y_train[:, mask])
    print("conjugate_gradient: SUCCESS!")
    test_score = pipe_3.score(X_test, Y_test[:, mask])
    print(f"Test R2 mean: {np.mean(backend.to_numpy(test_score)):.4f}")
except AssertionError as e:
    print(f"conjugate_gradient: FAILED with AssertionError - {e}")
except Exception as e:
    print(f"conjugate_gradient: FAILED - {type(e).__name__}: {e}")

print("\n=== Test Complete ===")
