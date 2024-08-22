from __future__ import annotations

import argparse
import os

from numpy import savetxt
from numpy.random import choice

from agents.agent_models.agent_tools import AgentTools
from agents.utils.paths import get_data_dir, get_test_data_dir


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

    return parser.parse_args()


def main():
    args = parse_args()

    num_records = args.num_records
    dataset_name = args.dataset_name

    data_dir = get_data_dir(__file__)
    test_data_dir = get_test_data_dir(__file__)
    dataset_path = os.path.join(test_data_dir, dataset_name + ".txt")
    if os.path.exists(dataset_path):
        raise ValueError("The file for the test data already exists: \n" + dataset_path)

    hdf5_file_path = os.path.join(data_dir, "wiki_forward.hdf5")
    matrix_info_path = os.path.join(data_dir, "matidx_sum_table.csv")
    summary_path = os.path.join(data_dir, "summary.csv")

    tools = AgentTools(hdf5_file_path, matrix_info_path, summary_path)

    source_target_ids = choice(tools.matidx_ary.max(), (num_records, 2), replace=False)

    savetxt(dataset_path, source_target_ids, delimiter="\t", fmt="%i")


if __name__ == "__main__":
    main()
