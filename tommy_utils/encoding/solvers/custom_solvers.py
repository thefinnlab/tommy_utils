import os
import warnings
import numbers

import numpy as np
import torch

from himalaya.backend import (
	get_backend,
	force_cpu_backend
)

from himalaya.backend._utils import _dtype_to_str
from himalaya.progress_bar import bar

from himalaya.scoring import l2_neg_loss
from himalaya.validation import (
	check_array,
	check_random_state,
	check_cv,
	_get_string_dtype
)

from himalaya.kernel_ridge import generate_dirichlet_samples
from himalaya.kernel_ridge._random_search import (
	_select_best_alphas,
	_decompose_kernel_ridge
)

from himalaya.ridge._random_search import _decompose_ridge

import time

from himalaya.ridge import (
	BandedRidgeCV
)

from himalaya.kernel_ridge import (
	MultipleKernelRidgeCV
)

######################################
### Custom banded ridge for group ####
######################################

class GroupLevelBandedRidge(BandedRidgeCV):

	@force_cpu_backend
	def fit(self, X, y=None):
		"""Fit the model.

		Parameters
		----------
		X : array of shape (n_samples, n_features), or list of length \
				(n_groups) with arrays of shape (n_samples, n_features)
			Training data.
			Must be a 2D array if ``groups`` is given.
			Must be a list of 2D arrays if ``groups="input"``.

		y : array of shape (n_samples,) or (n_samples, n_targets)
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""
		backend = get_backend()

		Xs = self._split_groups(X, check=True)
		del X

		self.n_features_in_ = sum(Xi.shape[1] for Xi in Xs)

		self.dtype_ = _get_string_dtype(Xs[0])
		device = "cpu" if self.Y_in_cpu else None
		y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)

		if any([np.mod(y.shape[0], Xi.shape[0]) for Xi in Xs]):
			raise ValueError("Inconsistent number of samples.")

		ravel = False
		if y.ndim == 1:
			y = y[:, None]
			ravel = True

		cv = check_cv(self.cv, y)

		# ------------------ call the solver
		tmp = self._call_solver(Xs=Xs, Y=y, cv=cv, return_weights=True,
								random_state=self.random_state,
								fit_intercept=self.fit_intercept,
								Y_in_cpu=self.Y_in_cpu)
		if self.fit_intercept:
			self.deltas_, self.coef_, self.cv_scores_ = tmp[:3]
			self.intercept_, = tmp[3:]
		else:
			self.deltas_, self.coef_, self.cv_scores_ = tmp

		if self.solver == "random_search":
			self.best_alphas_ = 1. / backend.exp(self.deltas_).sum(0)
		else:
			self.best_alphas_ = None

		if ravel:
			self.coef_ = self.coef_[:, 0]
			self.deltas_ = self.deltas_[:, 0]
			if self.fit_intercept:
				self.intercept_ = self.intercept_[0]

		return self

def solve_group_level_group_ridge_random_search(
	Xs, Y, n_samples_group, n_iter=100, concentration=[0.1,
									  1.0], alphas=1.0, fit_intercept=False,
	score_func=l2_neg_loss, cv=5, return_weights=False, local_alpha=True,
	jitter_alphas=False, random_state=None, n_targets_batch=None,
	n_targets_batch_refit=None, n_alphas_batch=None, progress_bar=True,
	conservative=False, Y_in_cpu=False, diagonalize_method="svd", warn=True):
	"""
	Customized by TLB for group-level encoding models. The assumption of this 
	solver is that each X_train contains the same features, while Y_train varies. 

	An example of this assumption is leave-one-subject-out models. Within each split,
	the solver averages across X_train and Y_train to predict the average Y_train. This
	is equivalent to solving across all Y_train simultaneously.

	Solve group ridge regression using random search on the simplex.

	Solve the group-regularized ridge regression::

		b* = argmin_b ||Z @ b - Y||^2 + ||b||^2

	where the feature space X_i is scaled by a group scaling ::

		Z_i = exp(deltas[i] / 2) X_i

	Parameters
	----------
	Xs : list of len (n_spaces), with arrays of shape (n_samples, n_features)
		Input features.
	Y : array of shape (n_samples, n_targets)
		Target data.
	n_samples_group: int, number of samples of unique Ys contained within Y.
		Used to split the data for averaging.
	n_iter : int, or array of shape (n_iter, n_spaces)
		Number of feature-space weights combination to search.
		If an array is given, the solver uses it as the list of weights
		to try, instead of sampling from a Dirichlet distribution.
	concentration : float, or list of float
		Concentration parameters of the Dirichlet distribution.
		If a list, iteratively cycle through the list.
		Not used if n_iter is an array.
	alphas : float or array of shape (n_alphas, )
		Range of ridge regularization parameter. The log group-weights
		``deltas`` are equal to log(gamma/alpha), where gamma is randomly
		sampled on the simplex, and alpha is selected from a list of
		candidates.
	fit_intercept : boolean
		Whether to fit an intercept.
		If False, Xs and Y must be zero-mean over samples.
	score_func : callable
		Function used to compute the score of predictions versus Y.
	cv : int or scikit-learn splitter
		Cross-validation splitter. If an int, KFold is used.
	return_weights : bool
		Whether to refit on the entire dataset and return the weights.
	local_alpha : bool
		If True, alphas are selected per target, else shared over all targets.
	jitter_alphas : bool
		If True, alphas range is slightly jittered for each gamma.
	random_state : int, or None
		Random generator seed. Use an int for deterministic search.
	n_targets_batch : int or None
		Size of the batch for over targets during cross-validation.
		Used for memory reasons. If None, uses all n_targets at once.
	n_targets_batch_refit : int or None
		Size of the batch for over targets during refit.
		Used for memory reasons. If None, uses all n_targets at once.
	n_alphas_batch : int or None
		Size of the batch for over alphas. Used for memory reasons.
		If None, uses all n_alphas at once.
	progress_bar : bool
		If True, display a progress bar over gammas.
	conservative : bool
		If True, when selecting the hyperparameter alpha, take the largest one
		that is less than one standard deviation away from the best.
		If False, take the best.
	Y_in_cpu : bool
		If True, keep the target values ``Y`` in CPU memory (slower).
	diagonalize_method : str in {"svd"}
		Method used to diagonalize the features.
	warn : bool
		If True, warn if the number of samples is smaller than the number of
		features.

	Returns
	-------
	deltas : array of shape (n_spaces, n_targets)
		Best log feature-space weights for each target.
	refit_weights : array of shape (n_features, n_targets), or None
		Refit regression weights on the entire dataset, using selected best
		hyperparameters. Refit weights are always stored on CPU memory.
	cv_scores : array of shape (n_iter, n_targets)
		Cross-validation scores per iteration, averaged over splits, for the
		best alpha. Cross-validation scores will always be on CPU memory.
	intercept : array of shape (n_targets,)
		Intercept. Only returned when fit_intercept is True.
	"""
	backend = get_backend()

	if backend.name == 'numpy':
		backend.split = np.split
	# elif backend.name == 'cupy':
	#     backend.split = cupy.split
	elif backend.name == 'torch' or backend.name == 'torch_cuda':
		backend.split = torch.tensor_split

	n_spaces = len(Xs)

	if isinstance(n_iter, int):
		gammas = generate_dirichlet_samples(n_samples=n_iter,
											n_kernels=n_spaces,
											concentration=concentration,
											random_state=random_state)
		gammas[0] = 1 / n_spaces
	elif n_iter.ndim == 2:
		gammas = n_iter
		assert gammas.shape[1] == n_spaces
	else:
		raise ValueError("Unknown parameter n_iter=%r." % (n_iter, ))

	if isinstance(alphas, numbers.Number) or alphas.ndim == 0:
		alphas = backend.ones_like(Y, shape=(1, )) * alphas

	dtype = Xs[0].dtype
	gammas = backend.asarray(gammas, dtype=dtype)
	device = getattr(gammas, "device", None)

	gammas, alphas = backend.check_arrays(gammas, alphas)


	# # TLB changing when we port items to GPU
	# Y = backend.asarray(Y, dtype=dtype, device="cpu" if Y_in_cpu else device)
	# Xs = [backend.asarray(X, dtype=dtype, device=device) for X in Xs]

	Y = backend.asarray(Y, dtype=dtype, device="cpu") # if Y_in_cpu else device)
	Xs = [backend.asarray(X, dtype=dtype, device="cpu") for X in Xs]

	# stack all features
	X_ = backend.concatenate(Xs, 1)
	n_features_list = [X.shape[1] for X in Xs]
	n_features = X_.shape[1]
	start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])

	slices = [
		slice(start, end)
		for start, end in zip(start_and_end[:-1], start_and_end[1:])
	]

	del Xs

	### TLB ADDING 
	# ensure all samples are same shape
	if np.mod(Y.shape[0], n_samples_group):
		raise ValueError("For group level models, all separate observations of Y "
						 "must have the same number of samples and features. This "
						 "means that Y.shape[0] should be divisible into n_samples_group.")

	n_samples, n_features = X_.shape

	# average the features here --> X is static across folds bc of group model
	X_all = backend.mean_float64(
				backend.stack(backend.split(X_, n_samples // n_samples_group)), axis=0) 

	Y_avg = backend.mean_float64(
				backend.stack(backend.split(Y, Y.shape[0] // n_samples_group)), axis=0) 

	X_all = backend.to_gpu(X_all)

	del X_

	if not Y_in_cpu:
		Y_avg = backend.to_gpu(Y_avg)

	# # average the features here
	# X_avg = backend.mean_float64(
	# 			backend.stack(backend.split(X_, n_samples_group)), axis=0) 

	if n_samples < n_features and warn:
		warnings.warn(
			"Solving banded ridge is slower than solving multiple-kernel ridge"
			f" when n_samples < n_features (here {n_samples} < {n_features}). "
			"Using linear kernels in "
			"himalaya.kernel_ridge.MultipleKernelRidgeCV or "
			"himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search "
			"would be faster. Use warn=False to silence this warning.",
			UserWarning)

	# if X_.shape[0] != Y.shape[0]:
	# 	raise ValueError("X and Y must have the same number of samples.")

	# X_offset, Y_offset = None, None

	# if fit_intercept:
	# 	X_offset = X_.mean(0)
	# 	Y_offset = Y.mean(0)
	# 	X_ = X_ - X_offset
	# 	Y = Y - Y_offset

	n_samples, n_targets = Y.shape

	if n_targets_batch is None:
		n_targets_batch = n_targets

	if n_targets_batch_refit is None:
		n_targets_batch_refit = n_targets_batch

	if n_alphas_batch is None:
		n_alphas_batch = len(alphas)

	cv = check_cv(cv, Y)
	n_splits = cv.get_n_splits()

	for train, val in cv.split(Y):
		if len(val) == 0 or len(train) == 0:
			raise ValueError("Empty train or validation set. "
							 "Check that `cv` is correctly defined.")

	random_generator, given_alphas = None, None

	if jitter_alphas:
		random_generator = check_random_state(random_state)
		given_alphas = backend.copy(alphas)

	best_gammas = backend.full_like(gammas, fill_value=1.0 / n_spaces,
									shape=(n_spaces, n_targets))

	best_alphas = backend.ones_like(gammas, shape=n_targets)

	cv_scores = backend.zeros_like(gammas, shape=(len(gammas), n_targets),
								   device="cpu")

	current_best_scores = backend.full_like(gammas, fill_value=-backend.inf,
											shape=n_targets)

	# initialize refit ridge weights
	refit_weights = None
	if return_weights:
		refit_weights = backend.zeros_like(gammas,
										   shape=(n_features, n_targets),
										   device="cpu")

	for ii, gamma in enumerate(
			bar(gammas, '%d random sampling with cv' % len(gammas),
				use_it=progress_bar)):
	
		for kk in range(n_spaces):
			X_device = getattr(X_all, "device", None)
			X_all[:, slices[kk]] *= backend.sqrt(backend.asarray(gamma[kk], device=X_device))

		if jitter_alphas:
			noise = backend.asarray_like(random_generator.rand(), alphas)
			alphas = given_alphas * (10 ** (noise - 0.5))

		scores = backend.zeros_like(gammas,
									shape=(n_splits, len(alphas), n_targets))

		for jj, (train, test) in enumerate(cv.split(Y)):
			# train = backend.to_gpu(train, device=device)
			# test = backend.to_gpu(test, device=device)

			# X_both = backend.mean_float64(
			# 	backend.stack(backend.split(X_, n_samples_group)), axis=0)

			# X_both = backend.to_gpu(X_both, device=device)

			# Xtrain = backend.mean_float64(
			# 	backend.stack(backend.split(X_[train], n_samples_group)), axis=0)
			# Xtest = backend.mean_float64(
			# 	backend.stack(backend.split(X_[test], n_samples_group)), axis=0)

			# Xtrain = backend.to_gpu(Xtrain, device=device)
			# Xtest = backend.to_gpu(Xtest, device=device)

			# if fit_intercept:
			# 	Xtrain_mean = X_[train].mean(0)
			# 	Xtrain = X_[train] - Xtrain_mean
			# 	Xtest = X_[test] - Xtrain_mean

			for matrix, alpha_batch in _decompose_ridge(
					Xtrain=X_all, alphas=alphas, negative_eigenvalues="nan",
					n_alphas_batch=n_alphas_batch, method=diagonalize_method):
				# n_alphas_batch, n_features, n_samples_train = \
				# matrix.shape
				matrix = backend.matmul(X_all, matrix)
				# n_alphas_batch, n_samples_test, n_samples_train = \
				# matrix.shape

				predictions = None

				for start in range(0, n_targets, n_targets_batch):
					batch = slice(start, start + n_targets_batch)

					Ytrain = Y[:, batch][train]
					Ytest = Y[:, batch][test]

					Ytrain = backend.mean_float64(
						backend.stack(backend.split(Ytrain, Ytrain.shape[0] // n_samples_group), axis=0), axis=0)
					Ytest = backend.mean_float64(
						backend.stack(backend.split(Ytest, Ytest.shape[0] // n_samples_group), axis=0), axis=0)

					Ytrain = backend.to_gpu(Ytrain, device=device)
					Ytest = backend.to_gpu(Ytest, device=device)

					# if fit_intercept:
					# 	Ytrain_mean = Ytrain.mean(0)
					# 	Ytrain = Ytrain - Ytrain_mean
					# 	Ytest = Ytest - Ytrain_mean

					predictions = backend.matmul(matrix, Ytrain)
					# n_alphas_batch, n_samples_test, n_targets_batch = \
					# predictions.shape

					with warnings.catch_warnings():
						warnings.filterwarnings("ignore", category=UserWarning)
						scores[jj, alpha_batch,
							   batch] = score_func(Ytest, predictions)
						# n_alphas_batch, n_targets_batch = score.shape
					del Ytrain, Ytest

				# make small alphas impossible to select
				too_small_alphas = backend.isnan(matrix[:, 0, 0])
				scores[jj, alpha_batch, :][too_small_alphas] = -1e5

				del matrix, predictions
			del train, test #, Xtrain, Xtest

		# select best alphas
		alphas_argmax, cv_scores_ii = _select_best_alphas(
			scores, alphas, local_alpha, conservative)

		cv_scores[ii, :] = backend.to_cpu(cv_scores_ii)

		# update best_gammas and best_alphas
		epsilon = np.finfo(_dtype_to_str(dtype)).eps
		mask = cv_scores_ii > current_best_scores + epsilon

		current_best_scores[mask] = cv_scores_ii[mask]

		best_gammas[:, mask] = gamma[:, None]
		best_alphas[mask] = alphas[alphas_argmax[mask]]

		# compute primal or dual weights on the entire dataset (nocv)
		if return_weights:
			update_indices = backend.flatnonzero(mask)

			if Y_in_cpu:
				update_indices = backend.to_cpu(update_indices)

			if len(update_indices) > 0:

				# refit weights only for alphas used by at least one target
				used_alphas = backend.unique(best_alphas[mask])

				primal_weights = backend.zeros_like(
					X_all, shape=(n_features, len(update_indices)), device="cpu")

				for matrix, alpha_batch in _decompose_ridge(
						Xtrain=backend.to_gpu(X_all, device=device), alphas=used_alphas,
						negative_eigenvalues="zeros",
						n_alphas_batch=min(len(used_alphas), n_alphas_batch),
						method=diagonalize_method):

					for start in range(0, len(update_indices),
									   n_targets_batch_refit):

						batch = slice(start, start + n_targets_batch_refit)

						weights = backend.matmul(
							matrix,
							backend.to_gpu(Y_avg[:, update_indices[batch]],
										   device=device))
						# used_n_alphas_batch, n_features, n_targets_batch = \
						# weights.shape

						# select alphas corresponding to best cv_score
						alphas_indices = backend.searchsorted(
							used_alphas, best_alphas[mask][batch])
						# mask targets whose selected alphas are outside the
						# alpha batch
						mask2 = backend.isin(
							alphas_indices,
							backend.arange(len(used_alphas))[alpha_batch])
						# get indices in alpha_batch
						alphas_indices = backend.searchsorted(
							backend.arange(len(used_alphas))[alpha_batch],
							alphas_indices[mask2])
						# update corresponding weights
						mask_target = backend.arange(weights.shape[2])
						mask_target = backend.to_gpu(mask_target)[mask2]
						tmp = weights[alphas_indices, :, mask_target]
						primal_weights[:, batch][:, backend.to_cpu(mask2)] = \
							backend.to_cpu(tmp).T

						del weights, alphas_indices, mask2, mask_target
					del matrix

				# multiply again by np.sqrt(g), as we then want to use
				# the primal weights on the unscaled features Xs, and not
				# on the scaled features (np.sqrt(g) * Xs)
				for kk in range(n_spaces):
					primal_weights[slices[kk]] *= backend.to_cpu(
						backend.sqrt(gamma[kk]))
				refit_weights[:, backend.to_cpu(mask)] = primal_weights
				del primal_weights

			del update_indices
		del mask

		for kk in range(n_spaces):
			X_all[:, slices[kk]] /= backend.sqrt(backend.asarray(gamma[kk], device=X_device))

	deltas = backend.log(best_gammas / best_alphas[None, :])

	if fit_intercept:
		intercept = (backend.to_cpu(Y_offset) -
					 backend.to_cpu(X_offset) @ refit_weights
					 ) if return_weights else None
		return deltas, refit_weights, cv_scores, intercept
	else:
		return deltas, refit_weights, cv_scores

######################################
### Custom kernel ridge for group ####
######################################

class GroupLevelMultipleKernelRidgeCV(MultipleKernelRidgeCV):
	"""Multiple-kernel ridge regression with cross-validation.

	Solve the kernel ridge regression::

		w* = argmin_w ||K @ w - Y||^2 + (w.T @ K @ w)

	where the kernel K is a weighted sum of multiple kernels Ks::

		K = sum_i exp(deltas[i]) Ks[i]

	The solver optimizes the log kernel weight ``deltas`` over
	cross-validation, using random search (``solver="random_search"``), or
	hyperparameter gradient descent (``solver="hyper_gradient"``).

	Parameters
	----------
	kernels : list of (str or callable), default=["linear", "polynomial"]
		List of kernel mapping. Available kernels are: 'linear',
		'polynomial, 'poly', 'rbf', 'sigmoid', 'cosine'.
		Set to 'precomputed' in order to pass a precomputed kernel matrix to
		the estimator methods instead of samples.
		A callable should accept two arguments and the keyword arguments passed
		to this object as kernel_params, and should return a floating point
		number.

	kernels_params : list of dict, or None
		Additional parameters for the kernel functions.
		See more details in the docstring of the function:
		``MultipleKernelRidgeCV.ALL_KERNELS[kernel]``

	solver : str
		Algorithm used during the fit, "random_search", or "hyper_gradient".

	solver_params : dict or None
		Additional parameters for the solver.
		See more details in the docstring of the function:
		``MultipleKernelRidgeCV.ALL_SOLVERS[solver]``

	fit_intercept : boolean
		Whether to fit an intercept.
		If False, X and Y must be zero-mean over samples.

	cv : int or scikit-learn splitter
		Cross-validation splitter. If an int, KFold is used.

	random_state : int, or None
		Random generator seed. Use an int for deterministic search.

	Y_in_cpu : bool
		If True, keep the target values ``y`` in CPU memory (slower).

	force_cpu : bool
		If True, computations will be performed on CPU, ignoring the
		current backend. If False, use the current backend.

	Attributes
	----------
	dual_coef_ : array of shape (n_samples) or (n_samples, n_targets)
		Representation of weight vectors in kernel space.

	intercept_ : float or array of shape (n_targets, )
		Intercept. Only present if fit_intercept is True.

	deltas_ : array of shape (n_kernels, n_targets)
		Log of kernel weights.

	cv_scores_ : array of shape (n_iter, n_targets)
		Cross-validation scores, averaged over splits.
		By default, the scores are computed with l2_neg_loss (in ]-inf, 0]).
		The scoring function can be changed with solver_params["score_func"].

	X_fit_ : array of shape (n_samples, n_features)
		Training data. If ``kernels == "precomputed"`` this is None.

	n_features_in_ : int
		Number of features (or number of samples if
		``kernels == "precomputed"``) used during the fit.

	dtype_ : str
		Dtype of input data.

	best_alphas_ : array of shape (n_targets, )
		Equal to ``1. / exp(self.deltas_).sum(0)``. For the "random_search"
		solver, it corresponds to the best hyperparameter alphas, assuming that
		each kernel weight vector sums to one (in particular, it is the case
		when ``solver_params['n_iter']`` is an integer).

	Examples
	--------
	>>> from himalaya.kernel_ridge import MultipleKernelRidgeCV
	>>> from himalaya.kernel_ridge import ColumnKernelizer
	>>> from himalaya.kernel_ridge import Kernelizer
	>>> from sklearn.pipeline import make_pipeline

	>>> # create a dataset
	>>> import numpy as np
	>>> n_samples, n_features, n_targets = 10, 5, 3
	>>> X = np.random.randn(n_samples, n_features)
	>>> Y = np.random.randn(n_samples, n_targets)

	>>> # Kernelize separately the first three columns and the last two
	>>> # columns, creating two kernels of shape (n_samples, n_samples).
	>>> ck = ColumnKernelizer(
	...     [("kernel_1", Kernelizer(kernel="linear"), [0, 1, 2]),
	...      ("kernel_2", Kernelizer(kernel="polynomial"), slice(3, 5))])

	>>> # A model with precomputed kernels, as output by ColumnKernelizer
	>>> model = MultipleKernelRidgeCV(kernels="precomputed")
	>>> pipe = make_pipeline(ck, model)
	>>> _ = pipe.fit(X, Y)
	"""

	@force_cpu_backend
	def fit(self, X, y=None, sample_weight=None):
		"""Fit the model.

		Parameters
		----------
		X : array of shape (n_samples, n_features)
			Training data. If kernels == "precomputed" this is instead
			a precomputed kernel array of shape
			(n_kernels, n_samples, n_samples).

		y : array of shape (n_samples,) or (n_samples, n_targets)
			Target values.

		sample_weight : None, or array of shape (n_samples, )
			Individual weights for each sample, ignored if None is passed.

		Returns
		-------
		self : returns an instance of self.
		"""
		backend = get_backend()

		ndim = 3 if self.kernels == "precomputed" else 2
		accept_sparse = False if self.kernels == "precomputed" else ("csr",
																	 "csc")
		X = check_array(X, accept_sparse=accept_sparse, ndim=ndim)
		self.dtype_ = _get_string_dtype(X)
		device = "cpu" if self.Y_in_cpu else None

		y = check_array(y, dtype=self.dtype_, ndim=[1, 2], device=device)

		n_samples = X.shape[1] if self.kernels == "precomputed" else X.shape[0]

		if any([np.mod(y.shape[0], Xi.shape[0]) for Xi in X]):
			raise ValueError("Inconsistent number of samples.")

		if sample_weight is not None:
			sample_weight = check_array(sample_weight, dtype=self.dtype_,
										ndim=1)
			if sample_weight.shape[0] != y.shape[0]:
				raise ValueError("Inconsistent number of samples.")

		Ks = self._get_kernels(X)

		self.X_fit_ = _to_cpu(X) if self.kernels != "precomputed" else None
		self.n_features_in_ = X.shape[-1]
		del X

		ravel = False
		if y.ndim == 1:
			y = y[:, None]
			ravel = True

		if sample_weight is not None:
			# We need to support sample_weight directly because K might be a
			# precomputed kernel.
			sw = backend.sqrt(sample_weight)[:, None]
			y = y * backend.to_cpu(sw) if self.Y_in_cpu else y * sw
			Ks *= (sw @ sw.T)[None]

		cv = check_cv(self.cv, y)

		# ------------------ call the solver
		tmp = self._call_solver(Ks=Ks, Y=y, cv=cv, return_weights="dual",
								Xs=None, random_state=self.random_state,
								fit_intercept=self.fit_intercept,
								Y_in_cpu=self.Y_in_cpu)

		# ------------------ store results
		self.deltas_, self.dual_coef_, self.cv_scores_ = tmp[:3]
		tmp = list(tmp[3:])

		if (self.solver_params is not None
				and self.solver_params.get("return_alphas", False)):
			self.best_alphas_ = tmp.pop(0)
		elif self.solver == "random_search":
			self.best_alphas_ = 1. / backend.exp(self.deltas_).sum(0)
		else:
			self.best_alphas_ = None

		if self.fit_intercept:
			self.intercept_ = tmp.pop(0)

		if ravel:
			self.dual_coef_ = self.dual_coef_[:, 0]
			self.deltas_ = self.deltas_[:, 0]

		return self

def solve_group_level_multiple_kernel_ridge_random_search(
		Ks, Y, n_samples_group, n_iter=100, concentration=[0.1, 1.0], alphas=1.0,
		score_func=l2_neg_loss, cv=5, fit_intercept=False, return_weights=None,
		Xs=None, local_alpha=True, jitter_alphas=False, random_state=None,
		n_targets_batch=None, n_targets_batch_refit=None, n_alphas_batch=None,
		progress_bar=True, Ks_in_cpu=False, conservative=False, Y_in_cpu=False,
		diagonalize_method="eigh", return_alphas=False):
	"""Solve multiple kernel ridge regression using random search.

	Parameters
	----------
	Ks : array of shape (n_kernels, n_samples, n_samples)
		Input kernels.
	Y : array of shape (n_samples, n_targets)
		Target data.
	n_samples_group: int, number of samples of unique Ys contained within Y.
		Used to split the data for averaging.
	n_iter : int, or array of shape (n_iter, n_kernels)
		Number of kernel weights combination to search.
		If an array is given, the solver uses it as the list of kernel weights
		to try, instead of sampling from a Dirichlet distribution. Examples:
		  - `n_iter=np.eye(n_kernels)` implement a winner-take-all strategy
			over kernels.
		  - `n_iter=np.ones((1, n_kernels))/n_kernels` solves a (standard)
			kernel ridge regression.
	concentration : float, or list of float
		Concentration parameters of the Dirichlet distribution.
		If a list, iteratively cycle through the list.
		Not used if n_iter is an array.
	alphas : float or array of shape (n_alphas, )
		Range of ridge regularization parameter.
	score_func : callable
		Function used to compute the score of predictions versus Y.
	cv : int or scikit-learn splitter
		Cross-validation splitter. If an int, KFold is used.
	fit_intercept : boolean
		Whether to fit an intercept. If False, Ks should be centered
		(see KernelCenterer), and Y must be zero-mean over samples.
		Only available if return_weights == 'dual'.
	return_weights : None, 'primal', or 'dual'
		Whether to refit on the entire dataset and return the weights.
	Xs : array of shape (n_kernels, n_samples, n_features) or None
		Necessary if return_weights == 'primal'.
	local_alpha : bool
		If True, alphas are selected per target, else shared over all targets.
	jitter_alphas : bool
		If True, alphas range is slightly jittered for each gamma.
	random_state : int, or None
		Random generator seed. Use an int for deterministic search.
	n_targets_batch : int or None
		Size of the batch for over targets during cross-validation.
		Used for memory reasons. If None, uses all n_targets at once.
	n_targets_batch_refit : int or None
		Size of the batch for over targets during refit.
		Used for memory reasons. If None, uses all n_targets at once.
	n_alphas_batch : int or None
		Size of the batch for over alphas. Used for memory reasons.
		If None, uses all n_alphas at once.
	progress_bar : bool
		If True, display a progress bar over gammas.
	Ks_in_cpu : bool
		If True, keep Ks in CPU memory to limit GPU memory (slower).
		This feature is not available through the scikit-learn API.
	conservative : bool
		If True, when selecting the hyperparameter alpha, take the largest one
		that is less than one standard deviation away from the best.
		If False, take the best.
	Y_in_cpu : bool
		If True, keep the target values ``Y`` in CPU memory (slower).
	diagonalize_method : str in {"eigh", "svd"}
		Method used to diagonalize the kernel.
	return_alphas : bool
		If True, return the best alpha value for each target.

	Returns
	-------
	deltas : array of shape (n_kernels, n_targets)
		Best log kernel weights for each target.
	refit_weights : array or None
		Refit regression weights on the entire dataset, using selected best
		hyperparameters. Refit weights are always stored on CPU memory.
		If return_weights == 'primal', shape is (n_features, n_targets),
		if return_weights == 'dual', shape is (n_samples, n_targets),
		else, None.
	cv_scores : array of shape (n_iter, n_targets)
		Cross-validation scores per iteration, averaged over splits, for the
		best alpha. Cross-validation scores will always be on CPU memory.
	best_alphas : array of shape (n_targets, )
		Best alpha value per target. Only returned if return_alphas is True.
	intercept : array of shape (n_targets,)
		Intercept. Only returned when fit_intercept is True.
	"""
	backend = get_backend()

	if backend.name == 'numpy':
		backend.split = np.split
	# elif backend.name == 'cupy':
	#     backend.split = cupy.split
	elif backend.name == 'torch' or backend.name == 'torch_cuda':
		backend.split = torch.tensor_split

	if isinstance(n_iter, int):
		gammas = generate_dirichlet_samples(n_samples=n_iter,
											n_kernels=len(Ks),
											concentration=concentration,
											random_state=random_state)
		gammas[0] = 1 / len(Ks)
	elif n_iter.ndim == 2:
		gammas = n_iter
		assert gammas.shape[1] == Ks.shape[0]
	else:
		raise ValueError("Unknown parameter n_iter=%r." % (n_iter, ))

	dtype = Ks.dtype
	gammas = backend.asarray(gammas, dtype=dtype)
	device = getattr(gammas, "device", None)

	# force y in cpu at fist before splitting later
	Y = backend.asarray(Y, dtype=dtype, device="cpu") # if Y_in_cpu else device)

	# split into subject-level samples here
	Y_avg = backend.mean_float64(
			backend.stack(backend.split(Y, Y.shape[0] // n_samples_group)), axis=0) 

	if not Y_in_cpu:
		Y_avg = backend.to_gpu(Y_avg)

	if isinstance(alphas, numbers.Number) or alphas.ndim == 0:
		alphas = backend.ones_like(Y, shape=(1, )) * alphas

	gammas, alphas, Xs = backend.check_arrays(gammas, alphas, Xs)

	Ks = backend.asarray(Ks, dtype=dtype,
						 device="cpu" if Ks_in_cpu else device)

	if fit_intercept:
		if return_weights == 'dual':
			Ks, Y, Ks_rows, Y_offset = _helper_intercept(Ks, Y)
		elif return_weights == 'primal':
			raise NotImplementedError(
				'Computing the intercept with return_weights="primal" is not'
				'implemented. Use fit_intercept=False or'
				'return_weights="dual".')
		else:
			raise ValueError(f'Cannot compute the intercept if return_weights'
							 f'is equal to {return_weights}.')

	# change this to be representative of the average shape
	n_samples, n_targets = Y_avg.shape

	if n_targets_batch is None:
		n_targets_batch = n_targets
	if n_targets_batch_refit is None:
		n_targets_batch_refit = n_targets_batch
	if n_alphas_batch is None:
		n_alphas_batch = len(alphas)

	cv = check_cv(cv, Y)
	n_splits = cv.get_n_splits()
	n_kernels = len(Ks)

	for train, val in cv.split(Y):
		if len(val) == 0 or len(train) == 0:
			raise ValueError("Empty train or validation set. "
							 "Check that `cv` is correctly defined.")

	if jitter_alphas:
		random_generator = check_random_state(random_state)
		given_alphas = backend.copy(alphas)

	best_gammas = backend.full_like(gammas, fill_value=1.0 / n_kernels,
									shape=(n_kernels, n_targets))
	best_alphas = backend.ones_like(gammas, shape=n_targets)
	cv_scores = backend.zeros_like(gammas, shape=(len(gammas), n_targets),
								   device="cpu")
	current_best_scores = backend.full_like(gammas, fill_value=-backend.inf,
											shape=n_targets)

	# initialize refit ridge weights
	if return_weights == 'primal':
		if Xs is None:
			raise ValueError("Xs is needed to compute the primal weights.")
		n_features = sum(X.shape[1] for X in Xs)
		refit_weights = backend.zeros_like(gammas,
										   shape=(n_features, n_targets),
										   device="cpu")
	elif return_weights == 'dual':
		refit_weights = backend.zeros_like(gammas,
										   shape=(n_samples, n_targets),
										   device="cpu")
	elif return_weights is None:
		refit_weights = None
	else:
		raise ValueError("Unknown parameter return_weights=%r." %
						 (return_weights, ))

	# Possibility to entirely skip the fit and return arbitrary weights.
	# This is to help designing integration smoke tests.
	if os.environ.get('HIMALAYA_SKIP_FIT', default=False):
		warnings.warn("Skip fit because HIMALAYA_SKIP_FIT=True.")
		# skip the loop by emptying the gammas candidates
		gammas = gammas[:0]
		# fill with fake weights, to avoid gettings only zeros.
		refit_weights += (backend.arange(n_targets)[None] + 1) / n_targets

	###########################################################################
	# Main loop over hyperparameter candidates (gammas)
	for ii, gamma in enumerate(
			bar(gammas, '%d random sampling with cv' % len(gammas),
				use_it=progress_bar)):

		if Ks_in_cpu:
			K = (backend.to_cpu(gamma[:, None, None]) * Ks).sum(0)
			K = backend.to_gpu(K, device=device)
		else:
			K = (gamma[:, None, None] * Ks).sum(0)

		if jitter_alphas:
			noise = backend.asarray_like(random_generator.rand(), alphas)
			alphas = given_alphas * (10 ** (noise - 0.5))

		scores = backend.zeros_like(gammas,
									shape=(n_splits, len(alphas), n_targets))

		for jj, (train, test) in enumerate(cv.split(Y)):

			# train = backend.to_gpu(train, device=device)
			# test = backend.to_gpu(test, device=device)

			# Ktrain, Ktest = K[train[:, None], train], K[test[:, None], train]

			# if fit_intercept:
			#     centerer = KernelCenterer()
			#     Ktrain = centerer.fit_transform(Ktrain)
			#     Ktest = centerer.transform(Ktest)

			for matrix, alpha_batch in _decompose_kernel_ridge(
					Ktrain=K, alphas=alphas, Ktest=K,
					negative_eigenvalues="nan", n_alphas_batch=n_alphas_batch,
					method=diagonalize_method):
				# n_alphas_batch, n_samples_test, n_samples_train = \
				# matrix.shape

				for start in range(0, n_targets, n_targets_batch):

					batch = slice(start, start + n_targets_batch)

					Ytrain = Y[:, batch][train]
					Ytest = Y[:, batch][test]

					Ytrain = backend.mean_float64(
						backend.stack(backend.split(Ytrain, Ytrain.shape[0] // n_samples_group), axis=0), axis=0)
					Ytest = backend.mean_float64(
						backend.stack(backend.split(Ytest, Ytest.shape[0] // n_samples_group), axis=0), axis=0)

					Ytrain = backend.to_gpu(Ytrain, device=device)
					Ytest = backend.to_gpu(Ytest, device=device)

					# Ybatch = backend.to_gpu(Y[:, batch], device=device)

					# Ytrain, Ytest = Ybatch[train], Ybatch[test]
					# del Ybatch

					# if fit_intercept:
					#     Ytrain_mean = Ytrain.mean(0)
					#     Ytrain = Ytrain - Ytrain_mean
					#     Ytest = Ytest - Ytrain_mean

					predictions = backend.matmul(matrix, Ytrain)
					# n_alphas_batch, n_samples_test, n_targets_batch = \
					# predictions.shape

					with warnings.catch_warnings():
						warnings.filterwarnings("ignore", category=UserWarning)
						scores[jj, alpha_batch, batch] = score_func(
							Ytest, predictions)
						# n_alphas_batch, n_targets_batch = score.shape
					del Ytrain, Ytest

				# make small alphas impossible to select
				too_small_alphas = backend.isnan(matrix[:, 0, 0])
				scores[jj, alpha_batch, :][too_small_alphas] = -1e5

				del matrix, predictions
			del train, test

		# select best alphas
		alphas_argmax, cv_scores_ii = _select_best_alphas(
			scores, alphas, local_alpha, conservative)
		cv_scores[ii, :] = backend.to_cpu(cv_scores_ii)

		# update best_gammas and best_alphas
		epsilon = np.finfo(_dtype_to_str(dtype)).eps
		if local_alpha:
			mask = cv_scores_ii > current_best_scores + epsilon
		else:
			# update based on score across all targets
			update = cv_scores_ii.mean() > current_best_scores.mean() + epsilon
			mask = backend.full_like(cv_scores_ii, fill_value=update,
									 dtype=bool)
		current_best_scores[mask] = cv_scores_ii[mask]
		best_gammas[:, mask] = gamma[:, None]
		best_alphas[mask] = alphas[alphas_argmax[mask]]

		# compute primal or dual weights on the entire dataset (nocv)
		if return_weights is not None:
			update_indices = backend.flatnonzero(mask)

			if Y_in_cpu:
				update_indices = backend.to_cpu(update_indices)

			if len(update_indices) > 0:

				# refit weights only for alphas used by at least one target
				used_alphas = backend.unique(best_alphas[mask])

				dual_weights = backend.zeros_like(
					K, shape=(n_samples, len(update_indices)), device="cpu")

				for matrix, alpha_batch in _decompose_kernel_ridge(
						K, used_alphas, Ktest=None,
						negative_eigenvalues="zeros",
						n_alphas_batch=min(len(used_alphas), n_alphas_batch),
						method=diagonalize_method):

					for start in range(0, len(update_indices),
									   n_targets_batch_refit):

						batch = slice(start, start + n_targets_batch_refit)

						weights = backend.matmul(
							matrix,
							backend.to_gpu(Y_avg[:, update_indices[batch]],
										   device=device))
						# used_n_alphas_batch, n_samples, n_targets_batch = \
						# weights.shape

						# select alphas corresponding to best cv_score
						alphas_indices = backend.searchsorted(
							used_alphas, best_alphas[mask][batch])

						# mask targets whose selected alphas are outside the
						# alpha batch
						mask2 = backend.isin(
							alphas_indices,
							backend.arange(len(used_alphas))[alpha_batch])

						# get indices in alpha_batch
						alphas_indices = backend.searchsorted(
							backend.arange(len(used_alphas))[alpha_batch],
							alphas_indices[mask2])

						# update corresponding weights
						mask_target = backend.arange(weights.shape[2])
						mask_target = backend.to_gpu(mask_target)[mask2]

						tmp = weights[alphas_indices, :, mask_target]

						dual_weights[:, batch][:, backend.to_cpu(mask2)] = \
							backend.to_cpu(tmp).T

						del weights, alphas_indices, mask2, mask_target
					del matrix

				if return_weights == 'primal':
					# multiply by g and not np.sqrt(g), as we then want to use
					# the primal weights on the unscaled features Xs, and not
					# on the scaled features (np.sqrt(g) * Xs)
					X = backend.concatenate([t * g for t, g in zip(Xs, gamma)],
											1)
					primal_weights = backend.to_cpu(X.T) @ dual_weights
					refit_weights[:, backend.to_cpu(mask)] = primal_weights
					
					del X, primal_weights

				elif return_weights == 'dual':
					refit_weights[:, backend.to_cpu(mask)] = dual_weights

				del dual_weights
			del update_indices
		del K, mask

	# End of main loop
	###########################################################################

	deltas = backend.log(best_gammas / best_alphas[None, :])
	if return_weights == 'dual':
		refit_weights *= backend.to_cpu(best_alphas)

	if fit_intercept:
		if return_weights == 'dual':
			intercept = backend.to_cpu(Y_offset)
			intercept -= ((backend.to_cpu(Ks_rows) @ refit_weights) *
						  backend.to_cpu(backend.exp(deltas))).sum(0)
		else:
			return NotImplementedError()

	results = [deltas, refit_weights, cv_scores]
	if return_alphas:
		results.append(best_alphas)
	if fit_intercept:
		results.append(intercept)
	return results


