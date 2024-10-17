from __future__ import annotations

import os
from argparse import ArgumentParser

import h5py
import numpy as np

from graph_prep import sampling, transforms


def main() -> None:
    parser = ArgumentParser(
        description="Generate a histogram of the number of steps between two nodes."
    )

    parser.add_argument("dataset_name", help="The name of the dataset", type=str)
    parser.add_argument(
        "--max_steps",
        help="The maximum number of steps to take",
        type=int,
        required=False,
        default=12,
    )
    parser.add_argument(
        "--total_paths",
        help="The total number of paths to generate",
        type=int,
        required=False,
        default=1000,
    )

    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    forward_link_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_forward_links.hdf5"
    )
    histogram_path = os.path.join(
        main_dir, "data/metadata", f"{args.dataset_name}_histogram.csv"
    )

    with h5py.File(forward_link_path, "r") as forw_f:
        forward_locs = forw_f["forward_locs"][:]
        forward_vals = forw_f["forward_vals"][:]

    backward_locs, backward_vals = transforms.swap_graph_direction(
        forward_locs, forward_vals, verbose=True
    )

    steps_list = sampling.generate_steps_list(
        args.total_paths,
        args.max_steps,
        forward_locs,
        forward_vals,
        backward_locs,
        backward_vals,
        verbose=True,
    )

    values = np.arange(args.max_steps)
    histogram = np.histogram(steps_list, bins=np.arange(args.max_steps + 1) - 0.5)[0]

    data_to_save = np.concatenate([values[:, None], histogram[:, None]], axis=1)

    np.savetxt(
        histogram_path,
        data_to_save,
        delimiter=",",
        header="Steps,Count",
        fmt="%d",
        comments="",
    )


if __name__ == "__main__":
    main()
