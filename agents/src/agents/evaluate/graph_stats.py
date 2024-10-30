from __future__ import annotations

import numpy as np
from tqdm import tqdm


# TAKEN FROM GRAPH PREP #
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
    forward_start = np.zeros(forward_vals.size + 1, dtype=int)

    for x in tqdm(forward_locs[1:, 0], desc="Read Links", disable=not verbose):
        forward_start[x] += 1

    forward_start = np.cumsum(forward_start[:-1])

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


def get_deadends(forward_locs: np.ndarray, give_array=False) -> int | np.ndarray:
    """
    Get the articles with no outgoing links.

    """
    deadends = forward_locs[:, 0] == forward_locs[:, 1]
    if give_array:
        return deadends

    return np.sum(deadends)


def get_unreacchable(
    forward_locs: np.ndarray, forward_vals: np.ndarray, give_array=False
) -> int | np.ndarray:
    """
    Get the articles that are not reachable from any page.
    """
    backward_locs, backward_vals = swap_graph_direction(forward_locs, forward_vals)
    unreachable = backward_locs[:, 0] == backward_locs[:, 1]
    if give_array:
        return unreachable

    return np.sum(unreachable)


def get_lone_articles(
    forward_locs: np.ndarray, forward_vals: np.ndarray, give_array=False
) -> int | np.ndarray:
    """
    Get the articles with no incoming or outgoing links.
    """
    deadends = get_deadends(forward_locs, give_array=True)
    unreachable = get_unreacchable(forward_locs, forward_vals, give_array=True)
    lone_articles = deadends * unreachable

    if give_array:
        return lone_articles

    return np.sum(lone_articles)
