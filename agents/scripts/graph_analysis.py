from __future__ import annotations

import argparse
import os

import pandas as pd

from agents.utils.data import get_data_folders, load_tools

GRAPH_NAMES = [
    # "en_wiki",
    # "fr_wiki",
    # "zh_wiki",
    # "ru_wiki",
    "bn_wiki",
    "oa_physics",
    # "oa_biochemistry",
    # "oa_medicine",
    # "oa_social_sciences",
    # "oa_arts",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get settings for analysing the graphs."
    )

    parser.add_argument(
        "-p",
        "--data_path",
        required=True,
        help="The path that the data are stored in, e.g. '/home/wiki-game/data/, \
            must include the 'data' folder",
    )

    return parser.parse_args()


def analysis():
    args = parse_args()
    data_path = args.data_path

    game_data_dir, _, _ = get_data_folders(data_path)

    for name in GRAPH_NAMES:
        print(name)

        game_data_path = os.path.join(game_data_dir, name, name + "_step_list.csv")
        from_database = False
        if name[:2] == "oa":
            from_database = True
            game_data_path = os.path.join(
                game_data_dir, name, name + "_step_list_lang.csv"
            )
        df = pd.read_csv(game_data_path)

        perc_total = (
            df[df["Steps"] < 1]["StartIndex"].count() / df["StartIndex"].count()
        )
        print("Percentage of total pairs games disconnected:", perc_total * 100)

        tools = load_tools(
            data_path, name, False, old_title_format=False, from_database=from_database
        )

        nodes_without_outward_links = []
        for idx, arr in enumerate(tools.forward_locs):
            if arr[0] == arr[1]:
                nodes_without_outward_links.append(idx)

        nodes_without_inward_links = [
            idx for idx in tools.matidx_ary if idx not in tools.forward_vals
        ]

        num_nodes = tools.matidx_ary.max()
        num_edges = len(tools.forward_vals)
        density = num_edges / (num_nodes * (num_nodes - 1))
        print("Total nodes: ", tools.matidx_ary.max())
        print("Total dead-end nodes: ", len(nodes_without_outward_links))
        print("Total unreachable nodes: ", len(nodes_without_inward_links))
        print("Total edges: ", len(tools.forward_vals))
        print("Directed density ", density)
        print()


if __name__ == "__main__":
    analysis()
