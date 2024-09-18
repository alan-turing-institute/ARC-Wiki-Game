from __future__ import annotations

import argparse
import os

import pandas as pd

from agents.utils.data import get_data_folders

"""
This script prints out some summary statistics about the output data. One of the
arguments to be set is the number of games to calculate the stats over, e.g., 100
or 10,000. If any output for a model doesn't have that number then it will be excluded
from the calculations. If any output has more than the number, then the first n rows
of output will be used.
Might want to update this to save the output somewhere.
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
    parser.add_argument(
        "--compare_over_n_games",
        required=True,
        type=int,
        help="The number of games to calculate the metrics over. Any output with fewer \
            than n games will not be included in the output.",
    )
    return parser.parse_args()


def evaluate(
    outputs: list[pd.DataFrame],
    input_dataset: str,
    experiment_name: str,
    model_name: str,
    n_games: int,
) -> dict[str | float] | None:
    df = pd.concat(outputs, axis=0)
    if df.shape[0] >= n_games:
        df = df[:n_games]

        sum_found_target = df["found_target"].sum()
        count_found_target = df["found_target"].count()
        perc_found_target = sum_found_target / count_found_target

        df_successful_games = df[df["found_target"]]
        median_num_steps = df_successful_games["num_steps"].median()
        mean_num_steps = df_successful_games["num_steps"].mean()
        df_successful_games["diff"] = (
            df_successful_games["num_steps"] - df_successful_games["optimal_steps"]
        )
        median_diff = df_successful_games["diff"].median()
        mean_diff = df_successful_games["diff"].mean()

        return {
            "input_dataset": input_dataset,
            "experiment_name": experiment_name,
            "model": model_name.rsplit("_", 1)[-1],
            "found_target_percentage": perc_found_target,
            "median_num_steps": median_num_steps,
            "mean_num_steps": mean_num_steps,
            "median_diff_to_optimal": median_diff,
            "mean_diff_to_optimal": mean_diff,
        }
    return None


def main():
    args = parse_args()
    data_path = args.data_path
    n_games = args.compare_over_n_games

    _, _, output_data_dir = get_data_folders(data_path)

    datasets = os.listdir(output_data_dir)

    evaluations = []

    for dataset in datasets:
        dataset_path = os.path.join(output_data_dir, dataset)
        if not os.path.isfile(dataset_path):
            output_folders = [
                folder
                for folder in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, folder))
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
                li_outputs = []
                prev_model_name = ""
                output_files.sort()
                for output_file in output_files:
                    file_path = os.path.join(dataset_path, output_folder, output_file)
                    model_name = output_file.rsplit("_", 1)[0]
                    if model_name != prev_model_name and len(prev_model_name) > 0:
                        evaluation = evaluate(
                            li_outputs,
                            input_dataset=dataset,
                            experiment_name=output_folder,
                            model_name=prev_model_name,
                            n_games=n_games,
                        )
                        if evaluation is not None:
                            evaluations.append(evaluation)
                        li_outputs = []
                    df = pd.read_csv(file_path, sep="\t")
                    li_outputs.append(df)
                    prev_model_name = model_name
                evaluation = evaluate(
                    li_outputs,
                    input_dataset=dataset,
                    experiment_name=output_folder,
                    model_name=prev_model_name,
                    n_games=n_games,
                )
                if evaluation is not None:
                    evaluations.append(evaluation)

    df_evaluations = pd.DataFrame(evaluations)
    print("Evaluations over " + str(n_games))
    print(df_evaluations)


if __name__ == "__main__":
    main()
