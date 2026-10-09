from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Get arguments for processing results")
    parser.add_argument(
        "config_file",
        type=str,
        help="The path of the config file to be used, relative to the agents package \
            e.g. 'result/wiki_embed_sum_config.yaml'",
    )
    return parser.parse_args()


def pull_game_ends(
    data_loc: str, number_list: list[int]
) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Pulls the information of how the game ends from the data files.

    Args:
        data_loc (str): The location of the data files.
        number_list (list[int]): The index of the files to pull.

    Returns:
        tuple: The number of games that ended in these certain ways:
            - Total games
            - Success
            - Hit the Step limit
            - Reached a Dead end
            - Hallucination by LLM
            - Bad request (GPT API rejection)
            - No response (Normally API timeout)
            - No answer (Reasoning model gave no answer, always 0 for other agents)
    """
    results_list = []

    for num in number_list:
        if not os.path.exists(data_loc + str(num) + ".csv"):
            # If this is reached then the file is missing!
            print("Can't find - " + data_loc + str(num) + ".csv")
            continue

        if os.stat(data_loc + str(num) + ".csv").st_size == 0:
            print("Empty file - " + data_loc + str(num) + ".csv")
            continue

        results_list.append(
            pd.read_csv(
                data_loc + str(num) + ".csv",
                sep="\t",
            )
        )

    if len(results_list) == 0:
        return (-1, -1, -1, -1, -1, -1, -1, -1)

    results_df = pd.concat(results_list)
    total_games = len(results_df)

    if (
        "error_message" in results_df.columns
        and results_df["error_message"].notna().any()
    ):
        hallucination_messages = (
            "Hallucination - title not in list",
            "No title provided in response",
            "Response not in valid JSON format",
            "Response not in JSON format",  # Older wording
        )

        results_df.loc[results_df["found_target"], "error_message"] = "Success"
        results_df["error_message"] = results_df["error_message"].fillna("No response")

        success_count = (results_df["error_message"] == "Success").sum()
        step_limit_count = (
            results_df["error_message"] == "Maximum number of steps reached"
        ).sum()
        dead_end_count = (
            results_df["error_message"] == "Ran out of titles to pick"
        ).sum()
        hallucination_count = (
            results_df["error_message"].isin(hallucination_messages).sum()
        )
        bad_request_count = (results_df["error_message"] == "BadRequestError").sum()
        no_response_count = (results_df["error_message"] == "No response").sum()
        # Only ReasoningAgent can produce this, so it is 0 for every other agent.
        no_answer_count = (
            results_df["error_message"] == "Response did not contain an answer"
        ).sum()
    else:
        success_count = (results_df["found_target"]).sum()
        step_limit_count = (results_df["num_steps"] == 100).sum()
        dead_end_count = total_games - success_count - step_limit_count
        hallucination_count = 0
        bad_request_count = 0
        no_response_count = 0
        no_answer_count = 0

    counted = (
        success_count
        + step_limit_count
        + dead_end_count
        + hallucination_count
        + bad_request_count
        + no_response_count
        + no_answer_count
    )
    if counted != total_games:
        msg = (
            f"Game outcomes do not sum to the total for {data_loc}: "
            f"{counted} counted vs {total_games} games. "
            "An error_message value is not being classified."
        )
        raise ValueError(msg)

    return (
        total_games,
        success_count,
        step_limit_count,
        dead_end_count,
        hallucination_count,
        bad_request_count,
        no_response_count,
        no_answer_count,
    )


def process_model(
    model_info: dict,
    main_dir: str,
    data_set_info: dict,
    stats_dict_list: list[dict],
) -> None:
    # Set up the paths and information
    number_list = model_info["file_index_list"]
    data_loc = os.path.join(
        main_dir,
        data_set_info["data_loc"],
        model_info["data_name"],
    )

    # Get the data
    (
        total_games,
        success_count,
        step_limit_count,
        dead_end_count,
        hallucination_count,
        bad_request_count,
        no_response_count,
        no_answer_count,
    ) = pull_game_ends(data_loc, number_list)

    # Save the statistics to the list
    stats_dict_list.append(
        {
            "Model": model_info["model_name"],
            "Language": data_set_info["language"],
            "Topic": data_set_info["topic"],
            "Total_games": total_games,
            "Success": success_count,
            "Step_limit": step_limit_count,
            "Dead_ends": dead_end_count,
            "Hallucination": hallucination_count,
            "Bad_requests": bad_request_count,
            "No_response": no_response_count,
            "No_answer": no_answer_count,
        }
    )


def main() -> None:
    args = parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(main_dir, args.config_file)) as yaml_file:
        res_yaml = yaml.safe_load(yaml_file.read())

    result_figure_loc = os.path.join(main_dir, res_yaml["result_figure_loc"])
    os.makedirs(result_figure_loc, exist_ok=True)

    stats_file_loc = os.path.join(main_dir, res_yaml["result_data_file"])
    if stats_file_loc[-9:] != "stats.csv":
        msg = "The stats file location must end in 'stats.csv'"
        raise ValueError(msg)
    stats_file_loc = stats_file_loc[:-9] + "game_count.csv"

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
            process_model(model_info, main_dir, data_set_info, stats_dict_list)
            pbar.update(1)
            model_count += 1

    pbar.close()

    # Save the statistics to a csv file
    stats_df = pd.DataFrame(
        stats_dict_list,
        columns=[
            "Model",
            "Language",
            "Topic",
            "Total_games",
            "Success",
            "Step_limit",
            "Dead_ends",
            "Hallucination",
            "Bad_requests",
            "No_response",
            "No_answer",
        ],
    )
    stats_df.to_csv(stats_file_loc, sep="\t", index=False)


if __name__ == "__main__":
    main()
