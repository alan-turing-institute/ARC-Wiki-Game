from __future__ import annotations

import duckdb
import h5py
import numpy as np
import pytest


@pytest.fixture(scope="module")
def lormum_ipsum() -> str:
    return (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam vitae "
        "sagittis est. Aliquam ultrices, dui at molestie condimentum, elit turpis "
        "volutpat sapien, ac ultricies metus ipsum sit amet leo. Integer nulla nisl, "
        "aliquet convallis sagittis vel, varius at metus. Morbi eget lacus a dolor "
        "dapibus tincidunt eget eu quam. Maecenas eros mauris, pretium sit amet "
        "convallis nec, maximus ac tortor. Lorem ipsum dolor sit amet, consectetur "
        "adipiscing elit. Orci varius natoque penatibus et magnis dis parturient "
        "montes, nascetur ridiculus mus. Aliquam sed mauris pharetra, vehicula magna "
        "a, sodales sem. Nunc tincidunt augue sed ornare posuere. Suspendisse auctor "
        "augue at tortor condimentum venenatis. Curabitur fringilla efficitur dui, "
        "non volutpat diam maximus a. Nunc rhoncus lacinia mattis. Aliquam "
        "scelerisque ipsum a odio interdum aliquet ac ut felis."
    )


@pytest.fixture(scope="module")
def testing_dir(tmp_path_factory: pytest.fixture.tmp_path_factory) -> str:
    # Create a temporary directory
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def hdf5_file(testing_dir: str) -> str:
    hdf5_path = str(testing_dir / "test_data.hdf5")

    # Create a graph to save to file
    fives = np.arange(0, 101, 5)
    test_locs = np.array([fives[0:-1], fives[1:]]).T
    twenty = np.arange(0, 20, 1)
    test_vals = np.concatenate(
        [
            twenty,  # One step
            twenty[::2],  # Two steps
            twenty[1::2],
            twenty[::4],
            twenty[1::4],  # Four steps
            twenty[2::4],
            twenty[3::4],
            twenty[::5],  # Five steps
            twenty[1::5],
            twenty[2::5],
            twenty[3::5],
            twenty[4::5],
            twenty[2::],  # Offset by 2
            np.array([0, 1]),
        ]
    )

    # Create and save some data to the HDF5 file
    with h5py.File(hdf5_path, "w") as f:
        # Create a dataset in the file
        f.create_dataset("forward_vals", data=test_vals)
        f.create_dataset("forward_locs", data=test_locs)

    return hdf5_path


@pytest.fixture(scope="module")
def summary_data(testing_dir: str, lormum_ipsum: str) -> str:
    summary_path = str(testing_dir / "summary_data.csv")

    with open(summary_path, "w") as f:
        f.write("PageID\tMatrixIndex\tPageTitle\tSummary\n")
        for i in range(20):
            title = chr(65 + i) * 4
            f.write(f"{i*2}\t{i}\t{title}\t{title} {lormum_ipsum}\n")

    return summary_path


@pytest.fixture(scope="module")
def matidx_info(testing_dir: str, summary_data: str) -> str:
    offset_ary = np.zeros(20).astype(int)

    offset = 0  # Offset to keep track of the current position in the file in bytes
    with open(summary_data, "rb") as sum_f:
        offset += len(sum_f.readline())  # Skip the header
        for i in range(20):
            offset_ary[i] = offset
            offset += len(sum_f.readline())

    matidx_info_path = str(testing_dir / "matidx_info.csv")

    with open(matidx_info_path, "w") as f:
        f.write("PageID\tMatrixIndex\tOffset\tPageTitle\n")
        for i in range(20):
            f.write(f"{i*2}\t{i}\t{offset_ary[i]}\t{chr(65 + i) * 4}\n")

    return matidx_info_path


@pytest.fixture(scope="module")
def summary_database(testing_dir: str, lormum_ipsum: str) -> str:
    summary_db_path = str(testing_dir / "summary_data.ddb")

    conn = duckdb.connect(summary_db_path)

    # Create table
    conn.execute(
        """
        CREATE TABLE works (
            MatrixIndex BIGINT,
            PageID BIGINT,
            PageTitle STRING,
            Abstract STRING
        )
        """
    )

    # Insert some data
    for i in range(20):
        title = chr(65 + i) * 4
        conn.execute(
            """
            INSERT INTO works (MatrixIndex, PageID, PageTitle, Abstract)
            VALUES (?, ?, ?, ?)""",
            (i, i * 2, title, f"{title} {lormum_ipsum}"),
        )

    conn.close()

    return summary_db_path


@pytest.fixture(scope="module")
def step_data(testing_dir: str) -> str:
    step_data_path = str(testing_dir / "step_data_0.csv")
    step_data_path2 = str(testing_dir / "step_data_100.csv")

    optimal_steps = np.tile(np.arange(1, 6), 20)
    num_steps = np.exp(0.3 + np.arange(1, 101) / 24).astype(int) + optimal_steps
    found_target1 = np.concatenate([np.tile([True, True, False], 33), [True]])
    found_target2 = np.tile([True, False], 50)

    with open(step_data_path, "w") as f:
        f.write("optimal_steps\tnum_steps\tfound_target\n")
        for i in range(100):
            f.write(f"{optimal_steps[i]}\t{num_steps[i]}\t{found_target1[i]}\n")

    with open(step_data_path2, "w") as f:
        f.write("optimal_steps\tnum_steps\tfound_target\n")
        for i in range(100):
            f.write(f"{optimal_steps[i]}\t{num_steps[i]}\t{found_target2[i]}\n")

    return step_data_path
