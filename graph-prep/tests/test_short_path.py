from __future__ import annotations

import numpy as np

from graph_prep.short_path import (
    _make_step,
    path_one_way,
    steps_check_deadend,
    steps_one_way,
    steps_two_way,
)


def test__make_step(forward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals

    # Check the first link
    reached_vec = np.zeros(20, dtype=bool)
    reached_vec[0] = True

    _make_step(reached_vec, forward_locs, forward_vals)
    expected = np.zeros(20, dtype=bool)
    expected[0:5] = True

    assert np.all(reached_vec == expected)

    # Check the fifth link
    reached_vec = np.zeros(20, dtype=bool)
    reached_vec[4] = True

    _make_step(reached_vec, forward_locs, forward_vals)
    expected = np.zeros(20, dtype=bool)
    expected[0:10:2] = True

    assert np.all(reached_vec == expected)


def test_steps_one_way(forward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals

    # One step case
    steps = steps_one_way(0, 2, forward_locs, forward_vals)
    assert steps == 1

    # Two step case
    steps = steps_one_way(1, 4, forward_locs, forward_vals)
    assert steps == 2

    small_loc = np.array([[0, 1], [1, 2], [2, 5]])
    small_vals = np.array([1, 0, 0, 1, 2])

    # can't reach
    steps = steps_one_way(0, 2, small_loc, small_vals, max_steps=5)
    assert steps == 5


def test_steps_two_way(forward_locs_vals, backward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals
    backward_locs, backward_vals = backward_locs_vals

    # One step case
    steps = steps_two_way(
        0, 2, forward_locs, forward_vals, backward_locs, backward_vals
    )
    assert steps == 1

    # Two step case
    steps = steps_two_way(
        1, 4, forward_locs, forward_vals, backward_locs, backward_vals
    )
    assert steps == 2

    small_loc_f = np.array([[0, 1], [1, 2], [2, 5]])
    small_vals_f = np.array([1, 0, 0, 1, 2])
    small_loc_b = np.array([[0, 2], [2, 4], [4, 5]])
    small_vals_b = np.array([1, 2, 0, 2, 2])

    # can't reach
    steps = steps_two_way(
        0, 2, small_loc_f, small_vals_f, small_loc_b, small_vals_b, max_steps=5
    )
    assert steps == 5


def test_steps_check_deadend(forward_locs_vals, backward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals
    backward_locs, backward_vals = backward_locs_vals

    # One step case
    steps = steps_check_deadend(
        0, 2, forward_locs, forward_vals, backward_locs, backward_vals
    )
    assert steps == 1

    # Two step case
    steps = steps_check_deadend(
        1, 4, forward_locs, forward_vals, backward_locs, backward_vals
    )
    assert steps == 2

    small_loc_f = np.array([[0, 1], [1, 2], [2, 5]])
    small_vals_f = np.array([1, 0, 0, 1, 2])
    small_loc_b = np.array([[0, 2], [2, 4], [4, 5]])
    small_vals_b = np.array([1, 2, 0, 2, 2])

    # can't reach
    steps = steps_check_deadend(
        0, 2, small_loc_f, small_vals_f, small_loc_b, small_vals_b, max_steps=5
    )
    assert steps == 5


def test_path_one_way(forward_locs_vals, backward_locs_vals):
    forward_locs, forward_vals = forward_locs_vals
    backward_locs, backward_vals = backward_locs_vals

    # One step case
    path = path_one_way(0, 2, forward_locs, forward_vals, backward_locs, backward_vals)
    assert path == [0, 2]

    # Two step case
    path = path_one_way(1, 4, forward_locs, forward_vals, backward_locs, backward_vals)
    assert path == [1, 8, 4]

    small_loc_f = np.array([[0, 1], [1, 2], [2, 5]])
    small_vals_f = np.array([1, 0, 0, 1, 2])
    small_loc_b = np.array([[0, 2], [2, 4], [4, 5]])
    small_vals_b = np.array([1, 2, 0, 2, 2])

    # can't reach
    path = path_one_way(
        0, 2, small_loc_f, small_vals_f, small_loc_b, small_vals_b, max_steps=5
    )
    assert path is None
