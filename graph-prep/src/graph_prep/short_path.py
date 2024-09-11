from __future__ import annotations

import numpy as np

__all__ = ["steps_one_way", "steps_two_way"]


def _make_step(
    reached_vec: np.ndarray,
    step_locs: np.ndarray,
    step_vals: np.ndarray,
    new_vec: np.ndarray | None = None,
) -> None:
    """Make a step in the graph traversal. (Hidden)

    This function updates the reached_vec array to indicate which nodes have been
    reached after taking a step through the graph.

    Args:
        reached_vec (np.ndarray): Boolean array of which nodes have been reached.
        step_locs (np.ndarray): Array of start and end indices for each step.
        step_vals (np.ndarray): Array of destinations of all the steps.
        new_vec (np.ndarray, optional): Array to store the new reached nodes. If None,
            the input reached_vec is updated in place. Defaults to None.

    Returns:
        None
    """
    if new_vec is None:
        new_vec = reached_vec

    for idx in np.where(reached_vec)[0]:
        s_loc, e_loc = step_locs[idx]
        if s_loc >= step_vals.size or s_loc == e_loc:
            continue  # Skip if the link is a self-loop or out of bounds

        new_vec[step_vals[s_loc:e_loc]] = True


def steps_one_way(
    start_link: int,
    end_link: int,
    forward_locs: np.ndarray,
    forward_vals: np.ndarray,
    max_steps: int = 10,
) -> int:
    """Calculate the number of steps required to traverse from start_link to end_link.

    Args:
        start_link (int): Index of the starting link.
        end_link (int): Index of the ending link.
        forward_locs (np.ndarray): Array of start and end indices for each forward step.
        forward_vals (np.ndarray): Array of destinations of all the steps.
        max_steps (int, optional): Maximum number of steps allowed. Defaults to 10.

    Returns:
        int: Number of steps taken to reach the end link.
    """
    reached_vec = np.zeros(forward_locs.shape[0], dtype=bool)
    reached_vec[start_link] = True

    steps = 0

    while not reached_vec[end_link] and steps < max_steps:
        _make_step(reached_vec, forward_locs, forward_vals)
        steps += 1

    return steps


def steps_two_way(
    start_link: int,
    end_link: int,
    forward_locs: np.ndarray,
    forward_vals: np.ndarray,
    backward_locs: np.ndarray,
    backward_vals: np.ndarray,
    max_steps: int = 10,
) -> int:
    """Calculate the number of steps required to traverse from start_link to end_link.

    Args:
        start_link (int): Index of the starting link.
        end_link (int): Index of the ending link.
        forward_locs (np.ndarray): Array of start and end indices for each forward step.
        forward_vals (np.ndarray): Array of destinations of all the steps going forward.
        backward_locs (np.ndarray): Array of start and end indices for each backward
            step.
        backward_vals (np.ndarray): Array of destinations of all the steps going
            backwards.
        max_steps (int, optional): Maximum number of steps allowed. Defaults to 10.

    Returns:
        int: Number of steps taken to reach the end link.
    """
    reached_f_vec = np.zeros(forward_locs.shape[0], dtype=bool)
    reached_f_vec[start_link] = True
    reached_b_vec = np.zeros(backward_locs.shape[0], dtype=bool)
    reached_b_vec[end_link] = True

    steps = 0

    while not np.any(reached_f_vec & reached_b_vec) and steps < max_steps:
        if steps % 2 == 0:
            # Go Forwards
            _make_step(reached_f_vec, forward_locs, forward_vals)
        else:
            # Go Backwards
            _make_step(reached_b_vec, backward_locs, backward_vals)

        steps += 1

    return steps


def steps_check_deadend(
    start_link: int,
    end_link: int,
    forward_locs: np.ndarray,
    forward_vals: np.ndarray,
    backward_locs: np.ndarray,
    backward_vals: np.ndarray,
    max_steps: int = 10,
) -> int:
    """Calculate the number of steps required to traverse from start_link to end_link.

    Args:
        start_link (int): Index of the starting link.
        end_link (int): Index of the ending link.
        forward_locs (np.ndarray): Array of start and end indices for each forward step.
        forward_vals (np.ndarray): Array of destinations of all the steps going forward.
        backward_locs (np.ndarray): Array of start and end indices for each backward
            step.
        backward_vals (np.ndarray): Array of destinations of all the steps going
            backwards.
        max_steps (int, optional): Maximum number of steps allowed. Defaults to 10.

    Returns:
        int: Number of steps taken to reach the end link.
    """
    reached_f_vec = np.zeros(forward_locs.shape[0], dtype=bool)
    reached_f_vec[start_link] = True
    reached_b_vec = np.zeros(backward_locs.shape[0], dtype=bool)
    reached_b_vec[end_link] = True

    steps = 0

    count_f = [1, 1]
    count_b = [1, 1]

    while not np.any(reached_f_vec & reached_b_vec) and steps < max_steps:
        if steps % 2 == 0:
            # Go Forwards
            _make_step(reached_f_vec, forward_locs, forward_vals)
            count_f[1] = np.sum(reached_f_vec)
            if count_f[0] == count_f[1]:
                return max_steps
            count_f[0] = count_f[1]
        else:
            # Go Backwards
            _make_step(reached_b_vec, backward_locs, backward_vals)
            count_b[1] = np.sum(reached_b_vec)
            if count_b[0] == count_b[1]:
                return max_steps
            count_b[0] = count_b[1]

        steps += 1

    return steps


def path_one_way(
    start_link: int,
    end_link: int,
    forward_locs: np.ndarray,
    forward_vals: np.ndarray,
    backward_locs: np.ndarray,
    backward_vals: np.ndarray,
    max_steps: int = 10,
) -> list[int] | None:
    """Find a random shortest path from the start link to the end link in a graph.

    Args:
        start_link (int): The starting link.
        end_link (int): The ending link.
        forward_locs (np.ndarray): The locations of forward links.
        forward_vals (np.ndarray): The values of forward links.
        backward_locs (np.ndarray): The locations of backward links.
        backward_vals (np.ndarray): The values of backward links.
        max_steps (int, optional): The maximum number of steps to take. Defaults to 10.

    Returns:
        list[int] | None: The one-way path from the start link to the end link,
            or None if no path is found.
    """
    reached_vectors = np.zeros((max_steps + 1, forward_locs.shape[0]), dtype=bool)
    reached_vectors[0, end_link] = True

    steps = 0
    while np.all(~reached_vectors[:, start_link]) and steps < max_steps:
        _make_step(
            reached_vectors[steps],
            backward_locs,
            backward_vals,
            reached_vectors[steps + 1],
        )
        steps += 1

    if steps == max_steps:
        return None

    path = [start_link]
    current_link = start_link
    for i in range(steps - 1, -1, -1):
        valid_destinations = np.zeros(forward_locs.shape[0], dtype=bool)
        valid_destinations[
            forward_vals[forward_locs[current_link, 0] : forward_locs[current_link, 1]]
        ] = True
        next_link = int(
            np.random.choice(np.where(reached_vectors[i] & valid_destinations)[0])
        )
        path.append(next_link)
        current_link = next_link

    return path
