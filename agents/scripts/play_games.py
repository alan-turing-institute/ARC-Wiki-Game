from __future__ import annotations

import argparse
import os

import yaml
from numpy import loadtxt

from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.utils.data import load_tools
from agents.utils.paths import get_game_dir, get_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Get config file to run")

    parser.add_argument(
        "-d",
        "--input_dataset",
        required=True,
        help="The name of the input dataset\
            which the experiments are to be run",
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        required=False,
        help="The name of the experiment to run (optional). If not provided, all \
            experiments in the config file will run.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        help="The name of the model to run for the given experiment (optional). If not \
            provided, all models in the config file will run for the given experiment.",
    )
    parser.add_argument(
        "-l",
        "--load_data_to_ram",
        required=False,
        type=bool,
        default=True,
        help="True if data is to be loaded to RAM, False if it remains on hard drive",
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
    load_data_to_ram = args.load_data_to_ram

    config_path = os.path.join(
        get_game_dir(__file__), input_dataset_name, "config.yaml"
    )
    assert os.path.exists(config_path), f"Config file does not exist: {config_path}"
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    if args.experiment_name is not None:
        assert (
            args.experiment_name in config["experiments"]
        ), f"Experiment does not exist: {args.experiment_name}"
        experiments = [args.experiment_name]
        if args.model_name is not None:
            assert (
                args.model_name in config["experiments"][args.experiment_name]["models"]
            ), f"Model name does not exist in experiment {args.experiment_name}: \
                {args.model_name}"
            # set the models list later as it will change by experiment if not set as
            # an arg
    else:
        experiments = config["experiments"].keys()
        if args.model_name is not None:
            for experiment in config["experiments"]:
                assert (
                    args.model_name in config["experiments"][experiment]["models"]
                ), f"Model name does not exist in experiment {experiment}: \
                    {args.model_name}"

    output_path = os.path.join(get_output_dir(__file__), input_dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # old_title_format currently hard-coded here to True, will need to update
    # how this is set when we come to work with multiple datasets.
    tools = load_tools(load_data_to_ram, old_title_format=True)

    for experiment in experiments:
        experiment_name = experiment
        experiment_output_path = os.path.join(output_path, experiment_name)
        os.makedirs(experiment_output_path, exist_ok=True)
        for game_dataset in config["experiments"][experiment]["game_datasets"]:
            source_ids, target_ids = load_test_dataset(input_dataset_name, game_dataset)
            if args.model_name is not None:
                models = [args.model_name]
            else:
                models = config["experiments"][experiment]["models"]
            for model in models:
                output_filename = game_dataset + "_" + model
                # We only run this combination if an output file doesn't already exist
                if not os.path.exists(
                    os.path.join(experiment_output_path, output_filename + ".csv")
                ):
                    agent = GreedyEmbeddingAgent(
                        tools,
                        max_rounds=config["experiments"][experiment]["params"][
                            "max_rounds"
                        ],
                        model_name=model,
                    )
                    agent.play_games(
                        source_ids, target_ids, experiment_output_path, output_filename
                    )


if __name__ == "__main__":
    main()
