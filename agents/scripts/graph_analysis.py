from __future__ import annotations

import argparse
from statistics import median

import pandas as pd

from agents.evaluate import graph_stats
from agents.utils.data import load_tools

GRAPH_NAMES = {
    "en_wiki": {"dataset": "Wiki", "topic": "English"},
    "fr_wiki": {"dataset": "Wiki", "topic": "French"},
    "ru_wiki": {"dataset": "Wiki", "topic": "Russian"},
    "zh_wiki": {"dataset": "Wiki", "topic": "Chinese"},
    "bn_wiki": {"dataset": "Wiki", "topic": "Bengali"},
    "oa_physics": {"dataset": "OA", "topic": "Physics"},
    "oa_biochemistry": {"dataset": "OA", "topic": "Biochemistry"},
    "oa_medicine": {"dataset": "OA", "topic": "Medicine"},
    "oa_social_sciences": {"dataset": "OA", "topic": "Social Sci"},
    "oa_arts": {"dataset": "OA", "topic": "Arts"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get settings for analysing the graphs."
    )

    parser.add_argument(
        "-p",
        "--data_path",
        required=True,
        type=str,
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
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="The path that the stats are to be stored.",
    )

    return parser.parse_args()


def analysis():
    args = parse_args()
    data_path = args.data_path
    load_data_to_ram = args.load_data_to_ram
    output_path = args.output_path

    li_di_stats = []

    for name in GRAPH_NAMES:
        print(name)

        di_stats = {
            "Dataset": GRAPH_NAMES[name]["dataset"],
            "Topic": GRAPH_NAMES[name]["topic"],
        }

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
        di_stats["num_nodes"] = num_nodes

        # Get list of nodes that don't have any links into them - these could still
        # be used as a start node in a game (assuming they have an outward link)
        num_nodes_no_inward_links = graph_stats.get_unreachable(
            tools.forward_locs, tools.forward_vals, False
        )
        di_stats["num_unreachable"] = num_nodes_no_inward_links
        di_stats["perc_unreachable"] = 100 * num_nodes_no_inward_links / num_nodes

        # Get list of nodes that don't have any links out of them - these can never
        # be a start node, but they could still be reached in a game (assuming they
        # have an inward link)
        num_nodes_no_outward_links = graph_stats.get_deadends(tools.forward_locs, False)
        di_stats["num_deadends"] = num_nodes_no_outward_links
        di_stats["perc_deadends"] = 100 * num_nodes_no_outward_links / num_nodes

        # Are there any nodes that completely disconnected? These will never be used
        # in a game
        num_nodes_disconnected = graph_stats.get_lone_articles(
            tools.forward_locs, tools.forward_vals, False
        )
        di_stats["num_nodes_disconnected"] = num_nodes_disconnected
        di_stats["perc_nodes_disconnected"] = 100 * num_nodes_disconnected / num_nodes

        # Number of nodes in play is the total number of nodes, minus the disconnected
        # nodes as these can never be reached or be used to start a game.
        num_nodes_in_game = num_nodes - num_nodes_disconnected
        di_stats["num_nodes_in_games"] = num_nodes_in_game

        # Number of nodes that can be navigated in a game
        di_stats["num_reachable_nodes_in_games"] = num_nodes - num_nodes_no_inward_links

        di_stats["num_deadends_in_games"] = (
            num_nodes_no_outward_links - num_nodes_disconnected
        )
        di_stats["perc_deadends_in_games"] = (
            100 * di_stats["num_deadends_in_games"] / num_nodes_in_game
        )

        num_edges = len(tools.forward_vals)
        di_stats["num_edges"] = num_edges
        di_stats["mean_edges_per_node"] = num_edges / num_nodes
        di_stats["mean_edges_per_node_in_games"] = num_edges / num_nodes_in_game

        # Calculate the mean and median title length of the articles
        article_lengths = []
        for i in range(1, num_nodes + 1):
            article_length = len(str(tools.get_article_title(i)))
            article_lengths.append(article_length)
        di_stats["mean_article_len"] = sum(article_lengths) / num_nodes
        di_stats["median_article_len"] = median(article_lengths)

        li_di_stats.append(di_stats)

    df_stats = pd.DataFrame(li_di_stats)
    df_stats.to_csv(output_path + "graph_analysis.tsv", sep="\t")

    df_metadata = df_stats[["Dataset", "Topic", "num_nodes", "mean_edges_per_node"]]
    df_metadata.rename(
        columns={"num_nodes": "Article", "mean_edges_per_node": "LinkPage"},
        inplace=True,
    )
    df_metadata["Article"] = round(df_metadata["Article"] / 1000000, 2)
    df_metadata["LinkPage"] = round(df_metadata["LinkPage"], 1)
    df_metadata.to_csv(output_path + "datasest_metadata.csv")


if __name__ == "__main__":
    analysis()
