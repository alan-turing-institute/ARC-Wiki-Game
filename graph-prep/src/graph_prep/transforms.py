from __future__ import annotations

import numpy as np
from tqdm import tqdm


def build_link_pairs(forward_locs: np.ndarray, forward_vals: np.ndarray) -> np.ndarray:
    """Build link pairs based on forward locations and values.

    Args:
        forward_locs (ndarray): The link indices of the forward links.
        forward_vals (ndarray): The values of the forward links.

    Returns:
        np.ndarray: The link pairs as a 2D array, with the first column being the
            start node and the second column being the end node.

    """
    forward_start = np.zeros_like(forward_vals)

    for x in forward_locs[1:, 0]:
        forward_start[x] += 1

    forward_start = np.cumsum(forward_start)

    return np.concatenate([forward_start[:, None], forward_vals[:, None]], axis=1)


def swap_graph_direction(
    forward_locs: np.ndarray, forward_vals: np.ndarray, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Reverses the direction of a graph represented by forward links.

    This will also work for going from a backward graph to a forward graph.


    Args:
        forward_locs (ndarray): The link indices of the forward links.
        forward_vals (ndarray): The values of the forward links.
        verbose (bool, optional): Whether to display progress information.
            Defaults to False.

    Returns:
        tuple: A tuple containing the backward locations and values.

    """
    forward_start = np.zeros_like(forward_vals)

    for x in tqdm(forward_locs[0, 1:], desc="Read Links", disable=not verbose):
        forward_start[x] += 1

    forward_start = np.cumsum(forward_start)

    if verbose:
        print("Sorting Forward Links")
    sorted = np.argsort(forward_vals)
    backward_start = forward_vals[sorted]
    backward_vals = forward_start[sorted]

    backward_locs = np.full_like(forward_locs, -1, dtype=int)

    for i in tqdm(
        range(backward_locs.shape[0]), desc="Reindexing Links", disable=not verbose
    ):
        backward_locs[i, 0] = np.searchsorted(backward_start, i)
        backward_locs[i, 1] = np.searchsorted(backward_start, i + 1)

    return backward_locs, backward_vals
