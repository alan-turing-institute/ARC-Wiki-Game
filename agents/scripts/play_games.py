from __future__ import annotations

import argparse
import os

import yaml
from numpy import loadtxt

# from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.utils.paths import get_config_dir, get_test_data_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Get config file to run")

    parser.add_argument(
        "-c", "--config", required=True, help="The name of the config file to be run"
    )

    return parser.parse_args()


def load_test_dataset(config: dict) -> tuple[list[int], list[int]]:
    dataset_name = config["test_dataset"]
    test_data_dir = get_test_data_dir("__file__")
    source_target_ids = loadtxt(
        os.path.join(test_data_dir, dataset_name + ".txt"), dtype=int, delimiter="\t"
    )
    source_ids = [int(i) for i in source_target_ids[:, 0]]
    target_ids = [int(i) for i in source_target_ids[:, 1]]
    return source_ids, target_ids


def main():
    args = parse_args()
    config_file_name = args.config
    config_path = get_config_dir(__file__)
    with open(os.path.join(config_path, config_file_name + ".yaml")) as stream:
        config = yaml.safe_load(stream)
    source_ids, target_ids = load_test_dataset(config)


if __name__ == "__main__":
    main()
