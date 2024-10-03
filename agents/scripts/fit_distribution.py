from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import yaml
from scipy.stats import nbinom
from tqdm import tqdm

from agents.evaluate import step_hist

CONFIG_FILE = "result/wiki_embed_sum_config.yaml"


def plot_figure(
    figure_loc: str,
    title: str,
    step_count: np.ndarray[int],
    optimal_steps: np.ndarray[int],
    mean_result: tuple[float, float],
    std_result: tuple[float, float],
    n_result: tuple[float, float],
    p_result: tuple[float, float],
    success_frac: tuple[float, float],
) -> None:
    """This makes the histogram figure for the negative binomial fit.
    It also has the statistics for the data and the fit."""
    x_p = np.arange(0, 101, 1)
    bins = np.arange(0, 101, 1) + 0.5

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.hist(
        step_count - optimal_steps,
        bins=bins,
        density=True,
        color="skyblue",
        label="Game Data",
    )
    ax.plot(
        x_p,
        nbinom.pmf(x_p, n_result[0], p_result[0]),
        "k--",
        label="Negative Binomial Fit",
    )

    ax.set_xlabel("Number of steps above optimal")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_title(title)
    ax.legend(
        loc="upper right",
        title=(
            f" ~Steps Stats~\n"
            f"Mean: {mean_result[0]:.2f} ± {mean_result[1]:.2f}\n"
            f"Std: {std_result[0]:.2f} ± {std_result[1]:.2f}\n"
            f"Success rate = {success_frac[0]:.1%} ± {success_frac[1]:.1%}\n"
            f" ~Negative Binomial Fit~\n"
            f"n = {n_result[0]:.3f} ± {n_result[1]:.3f} steps\n"
            f"p = {p_result[0]:.2%} ± {p_result[1]:.2%}"
        ),
        frameon=False,
        framealpha=0.8,
    )

    fig.savefig(figure_loc + ".png", dpi=200, bbox_inches="tight")

    plt.close(fig)


def pull_data(
    data_loc: str, number_list: list[int]
) -> tuple[np.ndarray[int], np.ndarray[int], np.ndarray[bool], int]:
    """
    Pull the data from the csv files and return the for the successful paths the path
    length and optimal steps, and for all games if the target was found and the total
    number of games.

    Args:
        data_loc (str): The location of the data files.
        number_list (list[int]): The index of the files to pull.

    Returns:
        sucessful_paths (np.ndarray): The path length for the successful paths.
        min_steps (np.ndarray): The optimal steps for the successful paths.
        if_sucess (np.ndarray): If the target was found.
        total_games (int): The total number of games.
    """
    results_list = []

    for num in number_list:
        results_list.append(
            pd.read_csv(
                data_loc + str(num) + ".csv",
                sep="\t",
            )
        )

    results_df = pd.concat(results_list)
    sucessful_paths = results_df["num_steps"][(results_df["found_target"])].values
    min_steps = results_df["optimal_steps"][(results_df["found_target"])].values
    if_sucess = results_df["found_target"].values
    total_games = len(results_df)

    return sucessful_paths, min_steps, if_sucess, total_games


def process_model(
    model_info, main_dir, result_figure_loc, data_set_info, stats_dict_list
) -> None:
    # Set up the paths and information
    number_list = model_info["file_index_list"]
    data_loc = os.path.join(
        main_dir,
        data_set_info["data_loc"],
        model_info["data_name"],
    )

    # Get the data
    sucessful_paths, min_steps, if_sucess, total_games = pull_data(
        data_loc, number_list
    )

    # Do Statistics
    mean_result, std_result = step_hist.bootstrap_steps(sucessful_paths)
    sucess_frac = step_hist.faction_success(if_sucess)
    n_result, p_result = step_hist.fit_nbinom(sucessful_paths, min_steps)

    # Plot the figure
    figure_title = (
        f"{model_info['model_name']}\n"
        f"{data_set_info['figure_title']} - {total_games:_d} Games"
    )
    figure_loc = os.path.join(
        main_dir, result_figure_loc, model_info["figure_file_name"]
    )
    plot_figure(
        figure_loc,
        figure_title,
        sucessful_paths,
        min_steps,
        mean_result,
        std_result,
        n_result,
        p_result,
        sucess_frac,
    )

    # Save the statistics to the list
    stats_dict_list.append(
        {
            "Model": model_info["model_name"],
            "Language": data_set_info["language"],
            "Mean": mean_result[0],
            "Mean_std": mean_result[1],
            "Standard_dev": std_result[0],
            "Standard_dev_std": std_result[1],
            "Success_rate": sucess_frac[0],
            "Success_rate_std": sucess_frac[1],
            "n": n_result[0],
            "n_std": n_result[1],
            "p": p_result[0],
            "p_std": p_result[1],
            "Total_games": total_games,
        }
    )


def main() -> None:
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(main_dir, CONFIG_FILE)) as yaml_file:
        res_yaml = yaml.safe_load(yaml_file.read())

    result_figure_loc = os.path.join(main_dir, res_yaml["result_figure_loc"])
    os.makedirs(result_figure_loc, exist_ok=True)

    stats_file_loc = os.path.join(main_dir, res_yaml["result_data_file"])

    stats_dict_list: list[dict] = []

    # Process the models
    model_count = 0
    total_models = sum(
        [
            len(data_set_info["model_list"])
            for data_set_info in res_yaml["data_set_list"]
        ]
    )
    pbar = tqdm(total=total_models, desc="Processing Models")

    for data_set_info in res_yaml["data_set_list"]:
        for model_info in data_set_info["model_list"]:
            pbar.set_postfix_str(model_info["model_name"])
            process_model(
                model_info, main_dir, result_figure_loc, data_set_info, stats_dict_list
            )
            pbar.update(1)
            model_count += 1

    pbar.close()

    # Save the statistics to a csv file
    stats_df = pd.DataFrame(
        stats_dict_list,
        columns=[
            "Model",
            "Language",
            "Mean",
            "Mean_std",
            "Standard_dev",
            "Standard_dev_std",
            "Success_rate",
            "Success_rate_std",
            "n",
            "n_std",
            "p",
            "p_std",
            "Total_games",
        ],
    )
    stats_df.to_csv(stats_file_loc, sep="\t", index=False)


if __name__ == "__main__":
    main()
