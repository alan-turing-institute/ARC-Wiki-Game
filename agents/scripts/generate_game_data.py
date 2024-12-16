from __future__ import annotations

import argparse
import os

from numpy import arange, column_stack, savetxt
from numpy.random import choice

from agents.utils.data import get_data_folders, load_tools


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
        "--from_database",
        required=False,
        default=False,
        type=bool,
        help="If True, then the database versions of agent tools will be used.",
    )

    parser.add_argument(
        "--min_target_index",
        type=int,
        required=False,
        default=None,
        help="The minimum value of the target index",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path
    num_records = args.num_records
    dataset_name = args.dataset_name
    input_dataset_name = args.input_dataset
    load_data_to_ram = args.load_data_to_ram
    use_old_title = args.old_title
    from_database = args.from_database
    min_target_index = args.min_target_index

    game_data_dir, _, _ = get_data_folders(data_path)
    os.makedirs(os.path.join(game_data_dir, input_dataset_name), exist_ok=True)
    game_data_path = os.path.join(
        game_data_dir, input_dataset_name, dataset_name + ".csv"
    )
    if os.path.exists(game_data_path):  # To prevent accidently overriding results
        raise ValueError(
            "The file for the test data already exists: \n" + game_data_path
        )

    tools = load_tools(
        data_path, input_dataset_name, load_data_to_ram, use_old_title, from_database
    )

    if min_target_index is not None:
        if tools.matidx_ary.max() - min_target_index < num_records:
            msg = (
                "You are requesting more records than there are possible targets pairs."
            )
            raise ValueError(msg)
        first_column = choice(arange(1, min_target_index), num_records, replace=False)
        second_column = choice(
            arange(min_target_index, tools.matidx_ary.max() + 1),
            num_records,
            replace=False,
        )
        source_target_ids = column_stack((first_column, second_column))
    else:
        source_target_ids = choice(
            arange(1, tools.matidx_ary.max() + 1), (num_records, 2), replace=False
        )

    savetxt(game_data_path, source_target_ids, delimiter=",", fmt="%i")


if __name__ == "__main__":
    main()
