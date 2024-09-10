from __future__ import annotations

import argparse

from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.utils.data import load_tools


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get parameters for playing a single game"
    )

    parser.add_argument(
        "-a",
        "--agent",
        required=True,
        type=str,
        default="GreedyEmbeddingAgent",
        help="The name of the agent (class) to use",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="average_word_embeddings_glove.6B.300d",
        help="The name of the model (if required) to use. For current implementation,\
            this will need to be a valid name for a model in the 'sentence-transformer'\
            HuggingFace library",
    )
    parser.add_argument(
        "-r",
        "--max_rounds",
        required=True,
        type=int,
        default=20,
        help="The maximum number of rounds of the game that will be played",
    )
    parser.add_argument(
        "-s",
        "--source",
        required=True,
        help="Either the ID or the title of the source (start) page",
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Either the ID or the title of the target (end) page",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        help="If True, then the result of the game will be printed out",
    )
    parser.add_argument(
        "-f",
        "--folder-name",
        type=str,
        help="The name of the folder to save the output of the game (if required)",
    )
    parser.add_argument(
        "-n",
        "--file_name",
        type=str,
        help="The name of the file to save the output of the game (if required)",
    )
    parser.add_argument(
        "-d",
        "--load_data_to_ram",
        type=bool,
        help="True if data is to be loaded to RAM, False if it remains on hard drive",
    )
    parser.add_argument(
        "-p",
        "--data_path",
        required=True,
        help="The path that the data are stored in, e.g. '/home/wiki-game/data/, \
            must include the 'data' folder",
    )
    parser.add_argument(
        "--old_title",
        required=False,
        default=False,
        type=bool,
        help="When loading the wiki data, this can be used if the old title format is\
            to be used",
    )
    parser.add_argument(
        "-i",
        "--input_dataset",
        required=True,
        help="The name of the input dataset\
            which the experiments are to be run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    input_dataset_name = args.input_dataset
    model_name = args.model
    max_rounds = args.max_rounds
    source = args.source if not args.source.isdigit() else int(args.source)
    target = args.target if not args.target.isdigit() else int(args.target)
    verbose = args.verbose if args.verbose is not None else False
    load_data_to_ram = args.load_data_to_ram
    use_old_title = args.old_title

    tools = load_tools(data_path, input_dataset_name, load_data_to_ram, use_old_title)

    if args.agent == "GreedyEmbeddingAgent":
        agent = GreedyEmbeddingAgent(
            tools, max_rounds=max_rounds, model_name=model_name
        )
        agent.play_game(source, target, verbose=verbose)
        if args.folder_name is not None and args.file_name is not None:
            folder_name = args.folder_name
            file_name = args.file_name
            agent.save_games(folder_name=folder_name, file_name=file_name)
    else:
        raise ValueError(
            "Invalid name for agent provided: "
            + args.agent
            + "\nCurrently the only valid values are: `GreedyEmbeddingAgent`"
        )


if __name__ == "__main__":
    main()
