from __future__ import annotations

import numpy as np
import pytest

from graph_prep.transforms import swap_graph_direction


@pytest.fixture(scope="module")
def forward_locs_vals() -> tuple[np.ndarray, np.ndarray]:
    # Create a graph to use for testing
    fives = np.arange(0, 101, 5)
    test_locs = np.array([fives[0:-1], fives[1:]]).T
    twenty = np.arange(0, 20, 1)
    test_vals = np.concatenate(
        [
            twenty,  # One step
            twenty[::2],  # Two steps
            twenty[1::2],
            twenty[::4],  # Four steps
            twenty[1::4],
            twenty[2::4],
            twenty[3::4],
            twenty[::5],  # Five steps
            twenty[1::5],
            twenty[2::5],
            twenty[3::5],
            twenty[4::5],
            twenty[2::],  # Offset by 2
            twenty[:2:],
        ]
    )

    return test_locs, test_vals


@pytest.fixture(scope="module")
def backward_locs_vals(forward_locs_vals) -> tuple[np.ndarray, np.ndarray]:
    # Create a the equivalent backward graph
    return swap_graph_direction(*forward_locs_vals)
