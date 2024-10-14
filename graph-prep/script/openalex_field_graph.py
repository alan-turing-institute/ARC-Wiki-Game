from __future__ import annotations

import argparse
import os

import duckdb
import h5py
import numpy as np
from tqdm import tqdm

from graph_prep.transforms import build_forward_graph

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

extracted_database_path = os.path.join(main_dir, "data/inter/openalex_attempt.ddb")


def parse_args():
    parser = argparse.ArgumentParser(description="Set the field to be extracted")

    parser.add_argument(
        "-l",
        "--long_name",
        required=True,
        type=str,
        help="The name of the field to be extracted from the OpenAlex dataset.",
    )
    parser.add_argument(
        "-s",
        "--short_name",
        required=False,
        type=str,
        help="The name to be given to the dataset when saving it - if not provided, \
            the long name will be used.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sci_field = args.long_name
    sci_field_short = (
        args.long_name.lower() if args.short_name is None else args.short_name
    )

    new_field_database_path = os.path.join(
        main_dir, "data/output/oa_" + sci_field_short + "_info.ddb"
    )

    forward_links_path = os.path.join(
        main_dir, "data/output/oa_" + sci_field_short + "_forward_links.hdf5"
    )

    extracted_db = duckdb.connect(extracted_database_path, read_only=True)
    new_field_db = duckdb.connect(new_field_database_path, read_only=False)

    total_database_size = extracted_db.execute(
        "SELECT COUNT(openalex_id) FROM works"
    ).fetchone()[0]

    new_field_db.execute(
        """
        CREATE or REPLACE TABLE works (
            MatrixIndex BIGINT,
            OpenAlexID BIGINT,
            PageTitle STRING,
            Abstract STRING,
            Language STRING,
            referenced_works BIGINT[],
        )
    """
    )

    chuck_size = 1_000_000

    for chunk_i in tqdm(
        range((total_database_size // chuck_size) + 1),
        desc=f"Inserting {sci_field} Works",
    ):
        sci_field_df = extracted_db.execute(
            f"""
            WITH work_chunk AS (
                SELECT openalex_id, title, abstract, language, referenced_works, field
                FROM works
                LIMIT {chuck_size} OFFSET {chuck_size*chunk_i}

            )
            SELECT openalex_id, title, abstract, language, referenced_works
            FROM work_chunk
            WHERE field = '{sci_field}'
        """
        ).fetchdf()
        new_field_db.register("sci_field_df", sci_field_df)
        new_field_db.execute(
            """
            INSERT INTO works (
                OpenAlexID, PageTitle, Abstract, Language, referenced_works
            )
            SELECT openalex_id, title, abstract, language, referenced_works
            FROM sci_field_df
        """
        )

    del sci_field_df
    extracted_db.close()

    print("Sort the works")
    new_field_db.execute(
        """
        CREATE TABLE works_sorted AS SELECT * FROM works ORDER BY OpenAlexID;
        DROP TABLE works;
        ALTER TABLE works_sorted RENAME TO works;
    """
    )

    new_field_db.execute(
        """
        WITH NumberedWorks AS (
            SELECT OpenAlexID, row_number() OVER (ORDER BY OpenAlexID) AS rn
            FROM works
        )
        UPDATE works
        SET MatrixIndex = (
            SELECT rn FROM NumberedWorks
            WHERE NumberedWorks.OpenAlexID = works.OpenAlexID
        )
    """
    )

    total_works = new_field_db.execute(
        "SELECT COUNT(MatrixIndex) FROM works"
    ).fetchone()[0]
    print(f"Total Works {total_works:_d}")

    total_refs = new_field_db.execute(
        """
        SELECT COUNT(*)
        FROM works, UNNEST(referenced_works) AS all_refs
    """
    ).fetchone()[0]
    print(f"Total References {total_refs:_d}")

    openalex_id_array = new_field_db.execute(
        """
        SELECT OpenAlexID FROM works
    """
    ).fetchnumpy()["OpenAlexID"]

    ref_works_df = new_field_db.execute(
        """
        SELECT MatrixIndex, referenced_works FROM works
    """
    ).fetchdf()

    ref_array = np.zeros((total_refs, 2), dtype=np.int64)
    i_ref = 0
    for _, row in tqdm(
        ref_works_df.iterrows(), total=total_works, desc="Indexing Referenced Works"
    ):
        for ref in row["referenced_works"]:
            mat_id = np.searchsorted(openalex_id_array, ref)
            if mat_id == len(openalex_id_array) or openalex_id_array[mat_id] != ref:
                continue
            ref_array[i_ref, 0] = row["MatrixIndex"]
            ref_array[i_ref, 1] = mat_id + 1
            i_ref += 1

    ref_array = ref_array[:i_ref]
    print(f"Total Valid References {i_ref:_d}")

    ref_array = np.concatenate([ref_array, ref_array[:, [1, 0]]])

    print("Sorting References")
    ref_array = ref_array[ref_array[:, 0].argsort()]

    forward_locs, forward_vals = build_forward_graph(
        ref_array, max_index=ref_works_df["MatrixIndex"].max(), verbose=True
    )

    print("Saving to HDF5")
    with h5py.File(forward_links_path, "w") as forward_f:
        forward_f.create_dataset("forward_locs", data=forward_locs)
        forward_f.create_dataset("forward_vals", data=forward_vals)

    new_field_db.execute(
        """
        ALTER TABLE works DROP COLUMN referenced_works;
    """
    )

    new_field_db.close()
    print(f"Saved Database to {new_field_database_path}")


if __name__ == "__main__":
    main()
