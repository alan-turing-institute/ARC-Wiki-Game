from __future__ import annotations

import numpy as np

from graph_prep.transforms import build_link_pairs, swap_graph_direction


def test_build_link_pairs(forward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals
    link_pairs = build_link_pairs(forward_locs, forward_vals)

    # Check the shape of the link pairs
    assert link_pairs.shape == (forward_vals.shape[0], 2)
    # Check the first few link pairs
    assert np.all(link_pairs[0:5, 0] == np.array([0, 0, 0, 0, 0]))
    assert np.all(link_pairs[0:5, 1] == np.array([0, 1, 2, 3, 4]))
    # Check the pairs are in order
    assert np.all(link_pairs[::5, 0] == np.arange(20))


def test_swap_graph_direction(forward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals
    backward_locs, backward_vals = swap_graph_direction(forward_locs, forward_vals)

    # Check the shape of the backward locations and values
    assert backward_locs.shape == forward_locs.shape
    assert backward_vals.shape == forward_vals.shape
    # Check that the locations are in order
    assert np.all(np.diff(backward_locs, axis=0) >= 0)
    # Check that the locations give a valid range
    assert np.all(backward_locs[:, 0] <= backward_locs[:, 1])
