from __future__ import annotations

import argparse
import os

import yaml
from numpy import loadtxt

from agents.agent_models.agent import GreedyEmbeddingAgent, PromptAgent
from agents.agent_models.tools import AgentTools
from agents.utils.paths import get_game_dir, get_output_dir, get_wiki_data_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Get config file to run")

    parser.add_argument(
        "-d",
        "--input_dataset",
        required=True,
        help="The name of the input dataset\
            which the experiments are to be run",
    )

    return parser.parse_args()


def load_test_dataset(
    input_dataset_name, game_dataset_name
) -> tuple[list[int], list[int]]:
    game_data_dir = get_game_dir(__file__)
    game_data_path = os.path.join(
        game_data_dir, input_dataset_name, game_dataset_name + ".txt"
    )

    source_target_ids = loadtxt(game_data_path, dtype=int, delimiter="\t")
    source_ids = [int(i) for i in source_target_ids[:, 0]]
    target_ids = [int(i) for i in source_target_ids[:, 1]]
    return source_ids, target_ids


def main():
    args = parse_args()
    input_dataset_name = args.input_dataset

    output_path = os.path.join(get_output_dir(__file__), input_dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # If or when we introduce more datasets, we will need to make the input dataset
    # an argument that will determine which one is loaded.
    hdf5_file_path, matrix_info_path, summary_path = get_wiki_data_paths(__file__)
    tools = AgentTools(hdf5_file_path, matrix_info_path, summary_path)

    config_path = os.path.join(
        get_game_dir(__file__), input_dataset_name, "config.yaml"
    )
    assert os.path.exists(config_path), "Config file does not exist: " + config_path
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    for experiment in config["experiments"]:
        experiment_name = experiment
        experiment_output_path = os.path.join(output_path, experiment_name)
        os.makedirs(experiment_output_path, exist_ok=True)
        for game_dataset in config["experiments"][experiment]["game_datasets"]:
            source_ids, target_ids = load_test_dataset(input_dataset_name, game_dataset)
            source_ids = source_ids[:10]
            target_ids = target_ids[:10]
            for model in config["experiments"][experiment]["models"]:
                output_filename = game_dataset + "_" + model
                # We only run this combination if an output file doesn't already exist
                if not os.path.exists(
                    os.path.join(experiment_output_path, output_filename + ".csv")
                ):
                    if (
                        config["experiments"][experiment]["agent_class"]
                        == "GreedyEmbeddingAgent"
                    ):
                        agent = GreedyEmbeddingAgent(
                            tools,
                            max_rounds=config["experiments"][experiment]["params"][
                                "max_rounds"
                            ],
                            model_name=model,
                        )
                    elif (
                        config["experiments"][experiment]["agent_class"]
                        == "PromptAgent"
                    ):
                        agent = PromptAgent(
                            tools,
                            max_rounds=config["experiments"][experiment]["params"][
                                "max_rounds"
                            ],
                            model_name=model,
                            temperature=config["experiments"][experiment]["params"][
                                "temperature"
                            ],
                            manual_seed=config["experiments"][experiment]["params"][
                                "seed"
                            ],
                        )
                    agent.play_games(
                        source_ids, target_ids, experiment_output_path, output_filename
                    )


if __name__ == "__main__":
    main()
