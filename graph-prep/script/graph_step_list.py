from __future__ import annotations

import os
from argparse import ArgumentParser

import duckdb
import h5py
import numpy as np

from graph_prep import sampling, transforms


def main() -> None:
    parser = ArgumentParser(
        description="Generate a set of index pairs and the number of steps."
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
    parser.add_argument(
        "--add_language",
        help="Set to True if the language code is also to be extracted from the \
            database (for OpenAlex data only)",
        type=bool,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--min_target_index",
        type=int,
        required=False,
        default=None,
        help="The minimum value of the target index",
    )

    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    forward_link_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_forward_links.hdf5"
    )
    histogram_path = os.path.join(
        main_dir, "data/metadata", f"{args.dataset_name}_step_list.csv"
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
        include_index=True,
        min_target_index=args.min_target_index,
    )

    header = "StartIndex,EndIndex,Steps"
    fmt = "%d"
    if args.add_language:
        info_db_path = os.path.join(
            main_dir, "data/output", f"{args.dataset_name}_info.ddb"
        )
        with duckdb.connect(info_db_path, read_only=True) as info_db:
            languages = info_db.execute(
                """SELECT MatrixIndex, Language, PageTitle FROM works"""
            ).fetchdf()
        header = "StartIndex,EndIndex,Steps,StartLang,EndLang"
        fmt = "%d,%d,%d,%s,%s"
        new_steps_list = []
        for item in steps_list:
            source_id = item[0]
            target_id = item[1]
            source_lang = languages.loc[source_id]["Language"]
            target_lang = languages.loc[target_id]["Language"]
            new_steps_list.append(
                np.concatenate(
                    [
                        item,
                        np.array([source_lang, target_lang]),
                    ],
                    dtype=object,
                )
            )
        steps_list = new_steps_list

    np.savetxt(
        histogram_path,
        steps_list,
        delimiter=",",
        header=header,
        fmt=fmt,
        comments="",
    )


if __name__ == "__main__":
    main()
