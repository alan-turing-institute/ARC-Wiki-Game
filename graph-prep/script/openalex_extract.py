from __future__ import annotations

import glob
import gzip
import json
import os
import re
from threading import Thread, active_count
from time import sleep

import duckdb

from graph_prep.read_openalex import extract_json

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_dir = os.path.join(main_dir, "data", "inter", "openalex_attempt.ddb")
temp_dir = os.path.join(main_dir, "data", "temp")
os.makedirs(temp_dir, exist_ok=True)

MAX_THREADS = 40
SPLIT_SIZE = 20_000
INSERT_SIZE = 500

file_names = glob.glob(os.path.join(main_dir, "data/raw/openalex/*/part*.gz"))

mdatabase = duckdb.connect(database_dir)
mdatabase.execute(
    "CREATE or REPLACE TABLE works "
    "(openalex_id BIGINT, title STRING, citations INT, doi STRING, "
    "field STRING, language STRING, abstract STRING, referenced_works BIGINT[])"
)

insert_string = """
INSERT INTO works
    (openalex_id, title, citations, doi, field,
    language, abstract, referenced_works)
VALUES
"""


def extract_file(t_file: str, database: duckdb.DuckDBPyConnection) -> None:
    """This function extracts the json data and inserts it into the database.
    It is called by the threads to process the data in parallel."""
    re_strip_escapes = re.compile(r"[\\']")
    re_extract_id = re.compile(r"W(\d+)$")

    print(
        f"Starting thread for {os.path.basename(t_file)} "
        f"with {active_count():2d} active threads"
    )

    with (
        open(t_file) as temp_f,
        database.cursor() as local_con,
    ):
        insert_list = []
        w = 0
        for line in temp_f:
            # Get json data
            data = json.loads(line)
            inset_vals = extract_json(data, re_extract_id, re_strip_escapes)
            if inset_vals is None:
                continue
            insert_list.append(inset_vals)
            if len(insert_list) >= INSERT_SIZE:
                local_con.execute(insert_string + ",\n".join(insert_list) + ";")
                insert_list = []
            w += 1

        if len(insert_list) > 0:
            local_con.execute(insert_string + ",\n".join(insert_list) + ";")

    os.remove(t_file)
    print(f"Finished thread for {os.path.basename(t_file)} with {w:d} works")


def write_temp_file(
    file_name: str, i_count: int, uncomp_lines: list[str], if_end: bool
) -> str:
    """This function writes the split json data to a temporary file.
    The temporary filename is retuned to be used by the threads."""
    temp_file = os.path.join(
        temp_dir,
        os.path.basename(os.path.dirname(file_name))
        + "_"
        + os.path.basename(file_name)[:-3]
        + f"_{i_count:03d}.json",
    )
    if if_end:
        with open(temp_file, "w") as t_f:
            t_f.writelines(uncomp_lines[i_count * SPLIT_SIZE :])
    else:
        with open(temp_file, "w") as t_f:
            t_f.writelines(
                uncomp_lines[i_count * SPLIT_SIZE : (i_count + 1) * SPLIT_SIZE]
            )

    return temp_file


threads = []
i_thread = 0
for file_name in sorted(file_names):
    print(f"Unzipping {file_name}")
    with gzip.open(file_name, "rt") as gz_f:
        uncomp_lines = gz_f.readlines()
        if len(uncomp_lines) == 0:  # Empty file
            continue

        # Split file into 10k chunks with the last chunk also having the remainder
        split_count = len(uncomp_lines) // SPLIT_SIZE
        split_count = 0 if split_count == 0 else split_count - 1

        if split_count > 0:  # More than 20k lines
            for i_count in range(split_count):
                temp_file = write_temp_file(file_name, i_count, uncomp_lines, False)
                threads.append(
                    Thread(
                        target=extract_file,
                        args=(temp_file, mdatabase),
                        name=f"{temp_file[-14:-4]}",
                    )
                )

        temp_file = write_temp_file(file_name, split_count, uncomp_lines, True)
        threads.append(
            Thread(
                target=extract_file,
                args=(temp_file, mdatabase),
                name=f"{temp_file[-14:-4]}",
            )
        )
    del uncomp_lines

    while len(threads) > i_thread:
        sleep(0.2)
        if active_count() < MAX_THREADS:
            threads[i_thread].start()
            i_thread += 1


for thread in threads:  # Wait for all threads to finish
    thread.join()

print("All threads finished")
print(
    f"Number of works: "
    f"{mdatabase.execute('SELECT COUNT(*) FROM works').fetchone()[0]:_d}"
)
print("Get fields info")
print(
    mdatabase.execute(
        """
    SELECT field,
        COUNT(*) AS field_count,
    FROM works
    GROUP BY field
    ORDER BY COUNT(*) DESC
"""
    ).df()
)
print("Get languages info (top 30)")
print(
    mdatabase.execute(
        """
        SELECT language,
            COUNT(*) AS language_count,
        FROM works
        GROUP BY language
        ORDER BY COUNT(*) DESC
        LIMIT 30
    """
    ).df()
)
print("All done")
