from __future__ import annotations

import csv
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Goes through the summary file and outputs the index table."
    )

    parser.add_argument("dataset_name", help="The name of the dataset.", type=str)

    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    title_info_tsv_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_title_info.tsv"
    )
    summary_tsv_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_summary.tsv"
    )
    matrix_index_info_tsv_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_matrix_index_info.tsv"
    )

    print(f"Reading title info from {title_info_tsv_path}")

    title_info_df = pd.read_csv(
        title_info_tsv_path, sep="\t", header=0, quoting=csv.QUOTE_NONE
    )

    # Make the empty arrays to store the new values
    article_count = (~title_info_df["IsRedirect"]).sum()

    page_id_ary = np.full(article_count, -1, dtype=int)
    matrix_id_ary = np.full(article_count, -2, dtype=int)
    page_title_ary = [""] * article_count
    offset_ary = np.full(article_count, -1, dtype=int)

    # Read the summary file and store the binary offset
    offset = 0
    with open(summary_tsv_path, "rb") as sum_f:
        offset += len(sum_f.readline())  # Skip header
        for n, line in enumerate(
            tqdm(sum_f, total=article_count, desc="Reading Summary")
        ):
            offset_ary[n] = offset
            offset += len(line)
            list_line = line.decode("utf-8").split("\t")
            page_id_ary[n] = int(list_line[0])
            matrix_id_ary[n] = int(list_line[1])
            page_title_ary[n] = list_line[2]

    if not (
        title_info_df["MatrixIndex"].max() == matrix_id_ary.max()
        or np.all(np.diff(matrix_id_ary) == 1)
        or matrix_id_ary[0] == 1
    ):
        msg = "Matrix Indexes are not ordered as expected."
        raise ValueError(msg)

    print("Creating the matrix index info table")
    matrix_index_info_df = pd.DataFrame(
        {
            "PageID": page_id_ary,
            "MatrixIndex": matrix_id_ary,
            "PageTitle": page_title_ary,
            "Offset": offset_ary,
        }
    )
    matrix_index_info_df.to_csv(
        matrix_index_info_tsv_path, sep="\t", header=True, index=False
    )

    print(f"Matrix Index Info table saved to {matrix_index_info_tsv_path}")


if __name__ == "__main__":
    main()
