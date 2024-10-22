from __future__ import annotations

import os
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def say_over(num, lc):
    total = lc.size
    over = sum(lc >= num)
    return f"Articles with {num:d}+ links:\n  {over/1000:.0f}k - {over/total:.1%}"


def main():
    parser = ArgumentParser(
        description=(
            "Looks at the graph and makes a histogram of the number of "
            "links in each article."
        )
    )

    parser.add_argument("dataset_name", help="The name of the dataset.", type=str)
    parser.add_argument("figure_title", help="The title of the figure.", type=str)

    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hdf5_dir = os.path.join(
        main_dir, f"data/output/{args.dataset_name}_forward_links.hdf5"
    )
    table_path = os.path.join(
        main_dir, f"data/metadata/link_count/{args.dataset_name}_link_count_table.csv"
    )
    figure_path = os.path.join(
        main_dir,
        f"data/metadata/link_count/{args.dataset_name}_link_count_histogram.png",
    )

    with h5py.File(hdf5_dir, "r") as h5file:
        locs = h5file["forward_locs"][:][:, 0]

    link_count = np.diff(locs)[:-1]

    # MAKE FIGURE #

    fig = plt.figure(figsize=(5, 4), dpi=200)
    ax = fig.add_subplot(111)

    ax.hist(
        link_count,
        bins=101,
        range=(0, 101),
        zorder=2,
        label="Link Data",
        color="blueviolet",
        alpha=0.9,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

    ax.set_xlabel("Number of links")
    ax.set_ylabel("Number of articles")
    ax.set_title("Distribution of number of links in articles\n" f"{args.figure_title}")
    ax.set_xlim(0, 100)
    ax.legend(
        title=say_over(100, link_count) + "\n" + say_over(500, link_count),
        loc="upper right",
        framealpha=1,
    )

    ax.grid(True, zorder=1)

    fig.savefig(figure_path, bbox_inches="tight")

    # MAKE TABLE #

    step_waypoints = [
        0,
        1,
        2,
        3,
        5,
        7,
        10,
        20,
        30,
        50,
        70,
        100,
        200,
        300,
        500,
        700,
        1000,
        2000,
        5000,
        10000,
        50000,
    ]

    stats_df = pd.DataFrame(
        {
            "step_number": step_waypoints,
            "larger_than": [sum(link_count >= way) for way in step_waypoints],
            "fraction": [
                sum(link_count >= way) / link_count.size for way in step_waypoints
            ],
        }
    )

    stats_df.to_csv(table_path, index=False)


if __name__ == "__main__":
    main()
