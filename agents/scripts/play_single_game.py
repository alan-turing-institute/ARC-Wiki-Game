from __future__ import annotations

import argparse

from agents.agent_models.agent import GreedyEmbeddingAgent, PromptAgent
from agents.agent_models.tools import AgentTools
from agents.utils.paths import get_wiki_data_paths


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
        default="all-distilroberta-v1",
        help="The name of the model (if required) to use. For current implementation,\
            this will need to be a valid name for a model in the 'sentence-transformer'\
            HuggingFace library",
    )
    parser.add_argument(
        "-r",
        "--max_rounds",
        required=True,
        type=int,
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

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    max_rounds = args.max_rounds
    source = args.source if not args.source.isdigit() else int(args.source)
    target = args.target if not args.target.isdigit() else int(args.target)
    verbose = args.verbose if args.verbose is not None else False

    # If or when we introduce more datasets, we will need to make the input dataset
    # an argument that will determine which one is loaded.
    hdf5_file_path, matrix_info_path, summary_path = get_wiki_data_paths(__file__)
    tools = AgentTools(hdf5_file_path, matrix_info_path, summary_path)

    if args.agent == "GreedyEmbeddingAgent":
        agent = GreedyEmbeddingAgent(
            tools, max_rounds=max_rounds, model_name=model_name
        )
    elif args.agent == "PromptAgent":
        agent = PromptAgent(tools, max_rounds=max_rounds, model_name=model_name)
    else:
        raise ValueError(
            "Invalid name for agent provided: "
            + args.agent
            + "\nCurrently the only valid values are: `GreedyEmbeddingAgent` or \
                `PromptAgent`"
        )
    agent.play_game(source, target, verbose=verbose)
    if args.folder_name is not None and args.file_name is not None:
        folder_name = args.folder_name
        file_name = args.file_name
        agent.save_games(folder_name=folder_name, file_name=file_name)


if __name__ == "__main__":
    main()
