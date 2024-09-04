from __future__ import annotations

import csv
import os
from argparse import ArgumentParser

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Goes through the list of links and builds the forward link graph."
    )

    parser.add_argument("dataset_name", help="The name of the dataset.", type=str)

    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    title_info_tsv_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_title_info.tsv"
    )
    link_data_path = os.path.join(
        main_dir, "data/inter", f"{args.dataset_name}_links.tsv"
    )
    forward_links_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_forward_links.hdf5"
    )

    print(f"Reading title info from {title_info_tsv_path}")
    title_info_df = pd.read_csv(
        title_info_tsv_path, sep="\t", header=0, quoting=csv.QUOTE_NONE
    )

    title_series = title_info_df["PageTitle"].astype(str)
    matrix_series = title_info_df["MatrixIndex"].astype(int)
    article_count = (~title_info_df["IsRedirect"]).sum()

    aprox_links = 0  # Approximation of the number of links for memory allocation
    with open(link_data_path) as link_f:
        for line in tqdm(link_f, total=article_count + 1, desc="Counting links"):
            aprox_links += 1 + line.count("|")

    link_data_array = np.full((aprox_links, 2), -1, dtype=int)
    link_count = 0
    with open(link_data_path) as link_f:
        link_f.readline()  # Skip header
        for line in tqdm(link_f, total=article_count, desc="Reading links"):
            line_list = line.split("\t")
            source_id = int(line_list[1])
            link_list = line_list[3].split("|")
            if len(link_list) == 0:  # A page with no links
                continue
            for link in link_list:
                title_loc = title_series.searchsorted(link)
                if (
                    len(title_series) < title_loc
                    or title_series.iloc[title_loc] != link
                ):  # If link not found
                    continue

                link_data_array[link_count, 0] = source_id
                link_data_array[link_count, 1] = matrix_series.iloc[title_loc]
                link_count += 1

    link_data_array = link_data_array[:link_count]

    if not (
        np.all(link_data_array[:, 0] >= 0)
        or np.all(np.diff(link_data_array[:, 0]) >= 0)
    ):
        msg = "The matrix index is not sorted like was expected"
        raise ValueError(msg)

    print(f"Found {link_count:_d} total links")

    max_matrix_index = matrix_series.max()
    forward_locs = np.full((max_matrix_index + 1, 2), -1, dtype=int)

    for i in tqdm(range(max_matrix_index), desc="Building forward locs"):
        forward_locs[i, 0] = np.searchsorted(link_data_array[:, 0], i)
        forward_locs[i, 1] = np.searchsorted(link_data_array[:, 0], i + 1)

    print("Saving to HDF5")
    with h5py.File(forward_links_path, "w") as forward_f:
        forward_f.create_dataset("forward_locs", data=forward_locs)
        forward_f.create_dataset("forward_vals", data=link_data_array[:, 1])

    print(f"Saved to {forward_links_path}")


if __name__ == "__main__":
    main()
