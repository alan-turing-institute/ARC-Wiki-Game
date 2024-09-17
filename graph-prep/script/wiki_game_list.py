from __future__ import annotations

import csv
import os

import pandas as pd
from tqdm import tqdm

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
step_list_path = os.path.join(main_dir, "data/metadata", "enwiki_step_list.csv")
matrix_info_path = os.path.join(main_dir, "data/output", "enwiki_matrix_index_info.tsv")
game_list_path = os.path.join(main_dir, "data/metadata", "enwiki_game_list.csv")

max_games = 2000  # The total number of games to generate
matrix_info_df = pd.read_csv(matrix_info_path, header=0, sep="\t")

with (
    open(step_list_path) as step_list_file,
    open(game_list_path, "w", newline="") as game_list_file,
):
    step_list_reader = csv.reader(step_list_file)
    game_list_writer = csv.writer(game_list_file)

    next(step_list_reader)  # Skip header
    # Write header in the game list
    game_list_writer.writerow(
        [
            "Num",
            "SpeedWikiURL",
            "TargetURL",
            "StartPage",
            "EndPage",
        ]
    )

    # Iterate through the step list
    count = 0
    for row in tqdm(step_list_reader, desc="Processing steps", total=max_games):
        start_index = int(row[0])
        end_index = int(row[1])

        if int(row[2]) == 0:
            continue

        # Get the corresponding matrix info
        start_title = matrix_info_df[matrix_info_df["MatrixIndex"] == start_index].iloc[
            0
        ]["PageTitle"]
        end_title = matrix_info_df[matrix_info_df["MatrixIndex"] == end_index].iloc[0][
            "PageTitle"
        ]

        # Encode the titles for URLs
        start_title_url = (
            start_title.replace(" ", "%20").replace("'", "%27").replace(",", "%2C")
        )
        end_title_url = (
            end_title.replace(" ", "%20").replace("'", "%27").replace(",", "%2C")
        )

        # Write the game list
        game_list_writer.writerow(
            [
                count,
                f"https://wikispeedrun.org/settings?"
                f"$=startingArticle$title={start_title_url};"
                f"&endingArticle$title={end_title_url};;",
                f"https://en.wikipedia.org/wiki/{end_title_url}",
                start_title,
                end_title,
            ]
        )

        # Increment the count
        count += 1
        if count > max_games:
            break
