from __future__ import annotations

import argparse
import os

import yaml
from numpy import array, loadtxt

from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.utils.data import get_data_folders, load_tools


def parse_args():
    parser = argparse.ArgumentParser(description="Get config file to run")

    parser.add_argument(
        "--input_dataset",
        required=True,
        help="The name of the input dataset which the experiments are to be run.\
             This is expected to match the name of a folder in the `input_data` folder",
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
            provided, then the code will check to see if a model id has been provided. \
            If neither are provided then all models in the config file will run for \
            the given experiment.",
    )
    parser.add_argument(
        "--model_id",
        required=False,
        type=int,
        help="The id of the model to run for the given experiment (optional). This \
            will only be used if the model name has not been provided. \
            If neither are provided then all models in the config file will run for \
            the given experiment.",
    )
    parser.add_argument(
        "--load_data_to_ram",
        required=False,
        type=bool,
        default=True,
        help="True if data is to be loaded to RAM, False if it remains on hard drive",
    )
    parser.add_argument(
        "--save_n_games",
        required=False,
        type=int,
        default=100,
        help="An integer that represents how often the output will be saved",
    )
    parser.add_argument(
        "-p",
        "--data_path",
        required=True,
        help="The path that the data are stored in, e.g. \
            '/home/ARC-WIKI-GAME/agents/data/, this must include the 'data' folder.\
             The 'data' folder is assumed to have three subfolders: 'game_data',\
             'input_data' and 'output_data'",
    )
    parser.add_argument(
        "--old_title",
        required=False,
        default=False,
        type=bool,
        help="When loading the wiki data, this can be used if the old title format is\
            to be used",
    )
    return parser.parse_args()


def load_test_dataset(
    input_dataset_name, game_data_dir, game_dataset_name
) -> tuple[list[int], list[int]]:
    game_data_path = os.path.join(
        game_data_dir, input_dataset_name, game_dataset_name + ".csv"
    )

    source_target_ids = loadtxt(game_data_path, dtype=int, delimiter=",", skiprows=1)
    steps = None
    if source_target_ids.shape[1] == 3:
        # The file also contains the number of steps, only load the valid games
        source_target_ids = array([row for row in source_target_ids if row[2] > 0])
        steps = [int(i) for i in source_target_ids[:, 2]]

    source_ids = [int(i) for i in source_target_ids[:, 0]]
    target_ids = [int(i) for i in source_target_ids[:, 1]]
    return source_ids, target_ids, steps


def main():
    args = parse_args()
    data_path = args.data_path
    input_dataset_name = args.input_dataset
    load_data_to_ram = args.load_data_to_ram
    save_n_games = args.save_n_games
    use_old_title = args.old_title

    game_data_dir, _, output_data_dir = get_data_folders(data_path)
    # Load config file
    config_path = os.path.join(game_data_dir, input_dataset_name, "config.yaml")
    if not os.path.exists(config_path):
        err_msg = f"Config file does not exist: {config_path}"
        raise ValueError(err_msg)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    # Set list of experiments to be run
    if args.experiment_name is not None:
        # Check the given experiment name exists in the config file
        if args.experiment_name not in config["experiments"]:
            err_msg = f"Experiment does not exist: {args.experiment_name}"
            raise ValueError(err_msg)
        experiments = [args.experiment_name]
    else:
        # Running all experiments in config file
        experiments = config["experiments"].keys()

    # If running for one specific model check that it exists in all experiments
    # that are to be run
    if args.model_name is not None:
        for experiment in experiments:
            if args.model_name not in config["experiments"][experiment]["models"]:
                err_msg = f"Model name does not exist in experiment {experiment}: \
                    {args.model_name}"
                raise ValueError(err_msg)
    elif args.model_id is not None:
        for experiment in experiments:
            if not args.model_id < len(config["experiments"][experiment]["models"]):
                err_msg = f"There are fewer models than the model id provided for \
                    experiment {experiment}"
                raise ValueError(err_msg)

    # Create output path if it doesn't already exist
    output_path = os.path.join(output_data_dir, input_dataset_name)
    os.makedirs(output_path, exist_ok=True)

    tools = load_tools(data_path, input_dataset_name, load_data_to_ram, use_old_title)

    for experiment in experiments:
        experiment_name = experiment
        experiment_output_path = os.path.join(output_path, experiment_name)
        os.makedirs(experiment_output_path, exist_ok=True)
        for game_dataset in config["experiments"][experiment]["game_datasets"]:
            source_ids, target_ids, steps = load_test_dataset(
                input_dataset_name, game_data_dir, game_dataset
            )
            # Create a list of model names to be run
            if args.model_name is not None:
                # One model name was specified in the config file
                models = [args.model_name]
            elif args.model_id is not None:
                models = [config["experiments"][experiment]["models"][args.model_id]]
            else:
                # No model specified in config, so run them all
                models = config["experiments"][experiment]["models"]
            for model in models:
                output_filename = game_dataset + "_" + model
                output_filename = output_filename.replace("/", "_")
                # We only run this combination if an output file doesn't already exist
                if not os.path.exists(
                    os.path.join(experiment_output_path, output_filename + ".csv")
                ):
                    # This logic will be updated when we have more than one agent type
                    agent = GreedyEmbeddingAgent(
                        tools,
                        max_rounds=config["experiments"][experiment]["params"][
                            "max_rounds"
                        ],
                        model_name=model,
                    )
                    agent.play_games(
                        source_ids,
                        target_ids,
                        experiment_output_path,
                        output_filename,
                        save_n_games,
                        optimal_steps=steps,
                    )


if __name__ == "__main__":
    main()
