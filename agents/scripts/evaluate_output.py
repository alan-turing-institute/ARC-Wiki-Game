from __future__ import annotations

import argparse
import os

import pandas as pd

from agents.utils.data import get_data_folders

"""
This whole script has been written very quickly and will need to be improved
We will want more functionality here - perhaps getting median / mean number
of steps per successful game and comparing to known shortest path algorithm.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Get arguments for evaluating output")
    parser.add_argument(
        "-p",
        "--data_path",
        required=True,
        help="The path that the data are stored in, e.g. '/home/wiki-game/data/, \
            must include the 'data' folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path

    _, _, output_data_dir = get_data_folders(data_path)

    datasets = os.listdir(output_data_dir)

    evaluations = []

    for dataset in datasets:
        dataset_path = os.path.join(output_data_dir, dataset)
        if not os.path.isfile(dataset_path):
            # get all output folders beginning with 'test' - this is just a quick
            # implementation for now to avoid evaluating other output files that
            # have been creating during debugging and early testing. Will probably
            # want to change this line or make the prefix name configurable.
            output_folders = [
                folder for folder in os.listdir(dataset_path) if folder[:4] == "test"
            ]
            for output_folder in output_folders:
                output_folder_path = os.path.join(dataset_path, output_folder)
                output_files = [
                    output_file
                    for output_file in os.listdir(output_folder_path)
                    if os.path.isfile(
                        os.path.join(dataset_path, output_folder, output_file)
                    )
                ]
                for output_file in output_files:
                    file_path = os.path.join(dataset_path, output_folder, output_file)
                    df = pd.read_csv(file_path, sep="\t")
                    sum_found_target = df["found_target"].sum()
                    count_found_target = df["found_target"].count()
                    perc_found_target = sum_found_target / count_found_target
                    # I know this is ugly, just want a quick way of getting the model
                    # name out of the file name.
                    model_name = output_file[:-4]
                    model_name = model_name[len(dataset) + len(output_folder) + 2 :]
                    evaluation = {
                        "input_dataset": dataset,
                        "test_dataset": output_folder,
                        "model": model_name,
                        "found_target_percentage": perc_found_target,
                    }
                    evaluations.append(evaluation)

    df_evaluations = pd.DataFrame(evaluations)
    print(df_evaluations)


if __name__ == "__main__":
    main()
