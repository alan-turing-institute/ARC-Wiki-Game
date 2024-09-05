from __future__ import annotations

import argparse
import os

from numpy import savetxt
from numpy.random import choice

from agents.utils.data import load_tools
from agents.utils.paths import get_game_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get settings for creating test dataset"
    )

    parser.add_argument(
        "-n",
        "--num_records",
        required=True,
        type=int,
        help="The number of records to be created.",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        required=True,
        help="The name of the dataset to be created",
    )
    parser.add_argument(
        "-i",
        "--input_dataset",
        required=True,
        help="The name of the input dataset to be used",
    )
    parser.add_argument(
        "-l",
        "--load_data_to_ram",
        type=bool,
        help="True if data is to be loaded to RAM, False if it remains on hard drive",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    num_records = args.num_records
    dataset_name = args.dataset_name
    input_dataset_name = args.input_dataset
    load_data_to_ram = args.load_data_to_ram

    assert input_dataset_name in [
        "wikipedia"
    ], "Invalid dataset name, currently only `wikipedia` is valid."

    game_data_dir = get_game_dir(__file__)
    os.makedirs(os.path.join(game_data_dir, input_dataset_name), exist_ok=True)
    game_data_path = os.path.join(
        game_data_dir, input_dataset_name, dataset_name + ".txt"
    )
    if os.path.exists(game_data_path):  # To prevent accidently overriding results
        raise ValueError(
            "The file for the test data already exists: \n" + game_data_path
        )

    # old_title_format currently hard-coded here to True, will need to update
    # how this is set when we come to work with multiple datasets.
    tools = load_tools(load_data_to_ram, old_title_format=True)

    source_target_ids = choice(tools.matidx_ary.max(), (num_records, 2), replace=False)

    savetxt(game_data_path, source_target_ids, delimiter="\t", fmt="%i")


if __name__ == "__main__":
    main()
