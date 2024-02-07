import os
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

############################
#### BRAINIAK FUNCTIONS ####
############################

def array_correlation(x, y, axis=0):

	"""Column- or row-wise Pearson correlation between two arrays

	Computes sample Pearson correlation between two 1D or 2D arrays (e.g.,
	two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
	between each corresponding column (axis=0) or row (axis=1) where axis
	indexes observations. If axis=0 (default), each column is considered to
	be a variable and each row is an observation; if axis=1, each row is a
	variable and each column is an observation (equivalent to transposing
	the input arrays). Input arrays must be the same shape with corresponding
	variables and observations. This is intended to be an efficient method
	for computing correlations between two corresponding arrays with many
	variables (e.g., many voxels).

	Parameters
	----------
	x : 1D or 2D ndarray
		Array of observations for one or more variables

	y : 1D or 2D ndarray
		Array of observations for one or more variables (same shape as x)

	axis : int (0 or 1), default: 0
		Correlation between columns (axis=0) or rows (axis=1)

	Returns
	-------
	r : float or 1D ndarray
		Pearson correlation values for input variables
	"""

	# Accommodate array-like inputs
	if not isinstance(x, np.ndarray):
		x = np.asarray(x)
	if not isinstance(y, np.ndarray):
		y = np.asarray(y)

	# Check that inputs are same shape
	if x.shape != y.shape:
		raise ValueError("Input arrays must be the same shape")

	# Transpose if axis=1 requested (to avoid broadcasting
	# issues introduced by switching axis in mean and sum)
	if axis == 1:
		x, y = x.T, y.T

	# Center (de-mean) input variables
	x_demean = x - np.mean(x, axis=0)
	y_demean = y - np.mean(y, axis=0)

	# Compute summed product of centered variables
	numerator = np.sum(x_demean * y_demean, axis=0)

	# Compute sum squared error
	denominator = np.sqrt(np.sum(x_demean ** 2, axis=0) *
						  np.sum(y_demean ** 2, axis=0))

##################################
##### STATISTICAL FUNCTIONS ######
##################################

from joblib import Parallel, delayed

def make_random_indices(n_items, n_perms, max_random_seed=2**32-1):
	
	random_state = None
	random_idxs = []
	
	for i in range(n_perms):
		if isinstance(random_state, np.random.RandomState):
			prng = random_state
		else:
			prng = np.random.RandomState(random_state)

		# get the permuted indices
		perm_idxs = prng.permutation(n_items)
		random_idxs.append(perm_idxs)

		#randomly select another random_state for next time
		random_state = np.random.RandomState(prng.randint(0, max_random_seed))
		
	return random_idxs

def block_permutation_test(true, pred, metric, block_size=10, n_perms=1000, N_PROC=1):
	'''
	Block permutation test of model predictions
	Adapted from https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/significance_testing.py
	'''
	
	# set the number of blocks based on size array --> get permutation indices
	n_blocks = int(true.shape[0] / block_size)
	perm_idxs = make_random_indices(n_items=n_blocks, n_perms=n_perms)
	
	# decompose into blocks
	block_true = np.dstack(np.vsplit(true, n_blocks)).transpose((2,0,1))
#     block_pred = np.dstack(np.vsplit(pred, n_blocks)).transpose((2,0,1))

	jobs = []
	for perm in perm_idxs:
		# create job for current iteration
		# permute true timeseries and compare with predicted
		job = delayed(metric)(np.vstack(block_true[perm, ...]), pred)
		jobs.append(job)

	with Parallel(n_jobs=N_PROC) as parallel:
		permutations = parallel(jobs)
		
	permutations = np.stack(permutations)
	
	return permutations

def p_from_null(observed, distribution, side='two-sided', exact=False, mult_comp_method=None, axis=None):
	"""Compute p-value from null distribution

	Returns the p-value for an observed test statistic given a null
	distribution. Performs either a 'two-sided' (i.e., two-tailed)
	test (default) or a one-sided (i.e., one-tailed) test for either the
	'left' or 'right' side. For an exact test (exact=True), does not adjust
	for the observed test statistic; otherwise, adjusts for observed
	test statistic (prevents p-values of zero). If a multidimensional
	distribution is provided, use axis argument to specify which axis indexes
	resampling iterations.

	The implementation is based on the work in [PhipsonSmyth2010]_.

	.. [PhipsonSmyth2010] "Permutation p-values should never be zero:
	   calculating exact p-values when permutations are randomly drawn.",
	   B. Phipson, G. K., Smyth, 2010, Statistical Applications in Genetics
	   and Molecular Biology, 9, 1544-6115.
	   https://doi.org/10.2202/1544-6115.1585

	Parameters
	----------
	observed : float
		Observed test statistic

	distribution : ndarray
		Null distribution of test statistic

	side : str, default: 'two-sided'
		Perform one-sided ('left' or 'right') or 'two-sided' test

	axis: None or int, default: None
		Axis indicating resampling iterations in input distribution

	Returns
	-------
	p : float
		p-value for observed test statistic based on null distribution
	"""

	if side not in ('two-sided', 'left', 'right'):
		raise ValueError("The value for 'side' must be either "
						 "'two-sided', 'left', or 'right', got {0}".
						 format(side))

	n_samples = len(distribution)
	
	if side == 'two-sided':
		# Numerator for two-sided test
		numerator = np.sum(np.abs(distribution) >= np.abs(observed), axis=axis)
	elif side == 'left':
		# Numerator for one-sided test in left tail
		numerator = np.sum(distribution <= observed, axis=axis)
	elif side == 'right':
		# Numerator for one-sided test in right tail
		numerator = np.sum(distribution >= observed, axis=axis)

	# If exact test all possible permutations and do not adjust
	if exact:
		p = numerator / n_samples

	# If not exact test, adjust number of samples to account for
	# observed statistic; prevents p-value from being zero
	else:
		p = (numerator + 1) / (n_samples + 1)

	if mult_comp_method:
		p = multipletests(p, method=mult_comp_method)[1]
	
	zvals = (observed.mean(axis) - distribution.mean(axis)) / distribution.std(axis)
	
	return zvals, p

# def p_from_null(observed, distribution, n_samples, mult_comp_method=None, stat_direction=None):
	
# 	if not stat_direction:
# 		numerator = np.sum(np.abs(distribution) >= np.abs(observed), axis=0) 
# 	else:
# 		numerator = np.sum(stat_direction(np.abs(distribution), np.abs(observed)), axis=0) 
	
# 	p = (numerator + 1) / (n_samples + 1)
	
# 	if len(p.shape) > 1:
# 		p = np.stack([multipletests(p[...,i], method='fdr_bh')[1] for i in range(p.shape[-1])], axis=-1)
# 	else:
# 		p = multipletests(p, method='fdr_bh')[1]
		
# 	return p

def pvalue_threshold(observed, p_values, alpha=0.05):

	observed_flat, p_flat = observed.flatten(), p_values.flatten()

	if np.argwhere(p_flat >= alpha).any():
		 observed_flat[np.argwhere(p_flat >= alpha)] = np.nan

	observed = observed_flat.reshape(observed.shape)

	return observed
