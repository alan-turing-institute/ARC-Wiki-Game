from __future__ import annotations

import argparse

from agents.utils.data import load_tools
from agents.evaluate import graph_stats, step_hist

GRAPH_NAMES = [
    "en_wiki",
    "fr_wiki",
    "zh_wiki",
    "ru_wiki",
    "bn_wiki",
    "oa_physics",
    "oa_biochemistry",
    "oa_medicine",
    "oa_social_sciences",
    "oa_arts",
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
    parser.add_argument(
        "--load_data_to_ram",
        required=False,
        default=True,
        type=bool,
        help="True if data is to be loaded to RAM, False if it remains on hard drive",
    )

    return parser.parse_args()


def analysis():
    args = parse_args()
    data_path = args.data_path
    load_data_to_ram = args.load_data_to_ram

    for name in GRAPH_NAMES:
        print(name)

        from_database = False
        if name[:2] == "oa":
            from_database = True

        tools = load_tools(
            data_path,
            name,
            load_data_to_ram,
            old_title_format=False,
            from_database=from_database,
        )

        num_nodes = tools.matidx_ary.max()
        print(f"Total nodes: {num_nodes:_d}", flush=True)

        # Get list of nodes that don't have any links into them - these could still
        # be used as a start node in a game (assuming they have an outward link)
        nodes_no_inward_links = [
            idx for idx in tools.matidx_ary if idx not in tools.forward_vals
        ]
        num_nodes_no_inward_links = len(nodes_no_inward_links)
        print(f"Total unreachable nodes: {num_nodes_no_inward_links:_d}")
        num_nodes_no_inward_links = graph_stats.get_unreacchable(tools.forward_locs, tools.forward_vals, False)
        print(f"Total unreachable nodes: {num_nodes_no_inward_links:_d}")

        # Get list of nodes that don't have any links out of them - these can never
        # be a start node, but they could still be reached in a game (assuming they
        # have an inward link)
        nodes_no_outward_links = []
        for idx, arr in enumerate(tools.forward_locs):
            if arr[0] == arr[1]:
                nodes_no_outward_links.append(idx)
        num_nodes_no_outward_links = len(nodes_no_outward_links)
        print(f"Total dead-end nodes: {num_nodes_no_outward_links:_d}")
        num_nodes_no_outward_links = graph_stats.get_deadends(tools.forward_locs, False)

        # Are there any nodes that completely disconnected? These will never be used
        # in a game
        nodes_disconnected = [
            idx for idx in nodes_no_inward_links if idx in nodes_no_outward_links
        ]
        num_nodes_disconnected = len(nodes_disconnected)
        print(f"Total disconnected nodes: {num_nodes_disconnected:_d}")

        # Number of nodes in play is the total number of nodes, minus the disconnected
        # nodes as these can never be reached or be used to start a game.
        num_nodes_in_game = num_nodes - num_nodes_disconnected
        print(f"Total nodes in game: {num_nodes_in_game:_d}")
        print(
            f"Total unreachable nodes in game: \
                {(num_nodes_no_inward_links - num_nodes_disconnected):_d}"
        )
        print(
            f"Total dead-end nodes in game: \
                {(num_nodes_no_outward_links-num_nodes_disconnected):_d}"
        )

        num_edges = len(tools.forward_vals)
        print(f"Total edges: {num_edges:_d}")

        # Calculated directed density, see https://en.wikipedia.org/wiki/Dense_graph
        # Using for both datasets, even though OpenAlex is symmetric - is this ok?
        density = num_edges / (num_nodes_in_game * (num_nodes_in_game - 1))
        print("Directed density ", density)
        print()


if __name__ == "__main__":
    analysis()
