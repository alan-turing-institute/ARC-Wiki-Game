from __future__ import annotations

import csv
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Reads the Metadata table and outputs the title table."
    )

    parser.add_argument("dataset_name", help="The name of the dataset.", type=str)

    args = parser.parse_args()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_tsv_path = os.path.join(
        main_dir, "data/inter", f"{args.dataset_name}_metadata.tsv"
    )
    title_info_tsv_path = os.path.join(
        main_dir, "data/output", f"{args.dataset_name}_title_info.tsv"
    )

    print(f"Reading metadata from {metadata_tsv_path}")
    metadata_df = pd.read_csv(
        metadata_tsv_path, sep="\t", header=0, quoting=csv.QUOTE_NONE
    )

    print("Organising metadata")
    page_id_ary = metadata_df["PageID"].values.astype(int)
    mat_index_ary = metadata_df["MatrixIndex"].values.astype(int)
    pagetitle_ser = metadata_df["PageTitle"].astype(str)
    redirect_ser = metadata_df["RedirectLocation"].astype(str)

    print("Sorting by the Page Titles")
    pagetitle_argsort = pagetitle_ser.argsort()
    sorted_page_id = page_id_ary[pagetitle_argsort]
    sorted_page_title_ser = pagetitle_ser[pagetitle_argsort]
    sorted_mat_index = mat_index_ary[pagetitle_argsort]
    sorted_redirect_ser = redirect_ser[pagetitle_argsort]

    # Make empty arrays to store the new values
    total_len = len(sorted_page_title_ser)

    new_page_id = np.full(total_len, -1, dtype=int)
    new_mat_index = np.full(total_len, -2, dtype=int)
    new_is_redirect = np.full(total_len, False, dtype=bool)
    new_page_title = [""] * total_len
    new_redirect = [""] * total_len

    i_r, i_r_temp = 0, 0

    for i in tqdm(range(total_len), desc="Finding the Redirects"):
        if sorted_mat_index[i] != -1:
            new_page_id[i] = sorted_page_id[i]
            new_mat_index[i] = sorted_mat_index[i]
            new_page_title[i] = sorted_page_title_ser.iloc[i]
            new_is_redirect[i] = False
            new_redirect[i] = sorted_page_title_ser.iloc[i]
        else:
            count_redirect = 0
            i_r = i
            while count_redirect < 10:  # Limit the number of redirects to 10
                count_redirect += 1
                i_r_temp = np.searchsorted(
                    sorted_page_title_ser, sorted_redirect_ser.iloc[i_r]
                )
                if (
                    i_r_temp > total_len
                    or sorted_page_title_ser.iloc[i_r_temp]
                    != sorted_redirect_ser.iloc[i_r]
                ):  # Redirect not found
                    i_r = None
                    break

                i_r = i_r_temp
                if sorted_mat_index[i_r] != -1:  # Found the final redirect
                    break

            if i_r is None or sorted_mat_index[i_r] == -1:
                continue

            new_page_id[i] = sorted_page_id[i]
            new_mat_index[i] = sorted_mat_index[i_r]
            new_page_title[i] = str(sorted_page_title_ser.iloc[i])
            new_is_redirect[i] = True
            new_redirect[i] = str(sorted_page_title_ser.iloc[i_r])

    print("Creating the Title Table")
    title_df = pd.DataFrame(
        {
            "PageID": new_page_id,
            "MatrixIndex": new_mat_index,
            "PageTitle": new_page_title,
            "IsRedirect": new_is_redirect,
            "RedirectLocation": new_redirect,
        }
    )

    # Remove the missing values
    title_df = title_df[title_df["MatrixIndex"] != -2]

    title_df.to_csv(title_info_tsv_path, sep="\t", index=False)

    print(f"Title Table saved to {title_info_tsv_path}")


if __name__ == "__main__":
    main()
