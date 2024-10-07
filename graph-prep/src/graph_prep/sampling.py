from __future__ import annotations

import numpy as np
from tqdm import tqdm

from graph_prep import short_path


def generate_steps_list(
    number_paths: int,
    max_steps: int,
    for_locs: np.ndarray[int],
    for_vals: np.ndarray[int],
    back_locs: np.ndarray[int],
    back_vals: np.ndarray[int],
    make_bad_zero: bool = True,
    verbose: bool = True,
    replacement: bool = False,
    seed: int | None = None,
    include_index: bool = False,
) -> np.ndarray[int]:
    """Generate a list of steps for multiple paths in a graph.

    Args:
        number_paths (int): The number of paths to generate.
        max_steps (int): The maximum number of steps allowed for each path.
        for_locs (np.ndarray[int]): The locations in the forward values for each index.
        for_vals (np.ndarray[int]): The destinations in the forward graph.
        back_locs (np.ndarray[int]): The locations in the backward values for each
            index.
        back_vals (np.ndarray[int]): The destinations in the backward graph.
        make_bad_zero (bool, optional): Whether to replace steps equal to max_steps
            with zero. Defaults to True.
        verbose (bool, optional): Whether to display progress information.
            Defaults to True.
        replacement (bool, optional): Whether to allow the same index to be chosen
            multiple times. Defaults to False.
        seed (int | None, optional): The seed to use for random number generation.

    Returns:
        np.ndarray[int]: The generated list of steps for each path.
    """
    rng = np.random.default_rng(seed)
    steps_list = []

    index_array = rng.choice(
        np.arange(0, for_locs.shape[0]),
        size=(number_paths, 2),
        replace=replacement,
    )

    for sp, tp in tqdm(
        index_array,
        desc="Generating Steps List",
        disable=not verbose,
    ):
        steps_list.append(
            short_path.steps_check_deadend(
                sp, tp, for_locs, for_vals, back_locs, back_vals, max_steps=max_steps
            )
        )

    steps_ary = np.array(steps_list, dtype=int)

    if make_bad_zero:
        steps_ary[steps_ary == max_steps] = 0

    if include_index:
        return np.concatenate([index_array, steps_ary[:, None]], axis=1)
    return steps_ary
