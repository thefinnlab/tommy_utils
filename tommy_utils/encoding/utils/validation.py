"""Cross-validation utilities for encoding models."""

import numpy as np
import itertools
from sklearn.utils.validation import check_random_state


def generate_leave_one_run_out(n_samples, run_onsets, random_state=None, n_runs_out=1):
    """Generate a leave-one-run-out split for cross-validation.

    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.
    random_state : None | int | instance of RandomState
        Random state for the shuffling operation.
    n_runs_out : int
        Number of runs to leave out in the validation set. Default to one.

    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
    """
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)

    if n_runs_out >= len(run_onsets):
        raise ValueError(
            "More runs requested for validation than there are "
            "total runs. Make sure that n_runs_out is less than "
            "than the number of runs (e.g., len(run_onsets))."
        )

    # Generate all combinations of runs for validation
    all_val_runs = np.array(list(itertools.combinations(range(n_runs), n_runs_out)))
    all_val_runs = random_state.permutation(all_val_runs)

    print(f'Total number of validation runs: {len(all_val_runs)}')

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])

    if any(len(run) == 0 for run in runs):
        raise ValueError(
            "Some runs have no samples. Check that run_onsets "
            "does not include any repeated index, nor the last "
            "index."
        )

    for val_runs in all_val_runs:
        train = [runs[jj] for jj in range(n_runs) if jj not in val_runs]
        val = [runs[jj] for jj in range(n_runs) if jj in val_runs]

        assert len(val) == n_runs_out  # Verify correct number of validation runs
        train, val = [np.hstack(x) for x in [train, val]]
        assert not np.isin(train, val).any()  # Ensure no overlap

        yield train, val
