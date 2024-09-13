from __future__ import annotations

import numpy as np

from graph_prep import sampling


def test_generate_steps_list(forward_locs_vals, backward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals
    backward_locs, backward_vals = backward_locs_vals

    # Test a basic set
    sample_dataset = sampling.generate_steps_list(
        50,
        8,
        forward_locs,
        forward_vals,
        backward_locs,
        backward_vals,
        verbose=False,
        replacement=True,
    )
    assert sample_dataset.shape == (50,)
    assert sample_dataset.dtype == np.int64
    assert np.all(sample_dataset >= 0)
    assert np.all(sample_dataset < 8)

    # Try the other options
    sample_dataset = sampling.generate_steps_list(
        5,
        8,
        forward_locs,
        forward_vals,
        backward_locs,
        backward_vals,
        verbose=False,
        replacement=False,
        make_bad_zero=False,
    )
    assert sample_dataset.shape == (5,)
    assert np.all(sample_dataset > 0)

    # Try with index
    sample_dataset = sampling.generate_steps_list(
        50,
        8,
        forward_locs,
        forward_vals,
        backward_locs,
        backward_vals,
        verbose=False,
        replacement=True,
        include_index=True,
    )
    assert sample_dataset.shape == (50, 3)
    assert sample_dataset.dtype == np.int64
    assert np.all(sample_dataset[:, 0] >= 0)
    assert np.all(sample_dataset[:, 0] < 20)
    assert np.all(sample_dataset[:, 2] < 8)
