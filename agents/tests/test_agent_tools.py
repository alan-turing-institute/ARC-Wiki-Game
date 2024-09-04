from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from agents.agent_tools import AgentToolsDisk, AgentToolsRAM

lormum_ipsum = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam vitae "
    "sagittis est. Aliquam ultrices, dui at molestie condimentum, elit turpis volutpat "
    "sapien, ac ultricies metus ipsum sit amet leo. Integer nulla nisl, aliquet "
    "convallis sagittis vel, varius at metus. Morbi eget lacus a dolor dapibus "
    "tincidunt eget eu quam. Maecenas eros mauris, pretium sit amet convallis nec, "
    "maximus ac tortor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Orci "
    "varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. "
    "Aliquam sed mauris pharetra, vehicula magna a, sodales sem. Nunc tincidunt augue "
    "sed ornare posuere. Suspendisse auctor augue at tortor condimentum venenatis. "
    "Curabitur fringilla efficitur dui, non volutpat diam maximus a. Nunc rhoncus "
    "lacinia mattis. Aliquam scelerisque ipsum a odio interdum aliquet ac ut felis."
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


def test_read_hdf5(hdf5_file: str):
    with h5py.File(hdf5_file, "r") as h5_file:
        dset = h5_file["forward_vals"]
        forward_vals = dset[:]
        dset = h5_file["forward_locs"]
        forward_locs = dset[:]

        assert np.all(forward_vals[0:5] == np.arange(0, 5, 1))
        assert np.all(forward_locs[:, 0] == np.arange(0, 96, 5))


@pytest.fixture(scope="module")
def summary_data(testing_dir: str) -> str:
    summary_path = str(testing_dir / "summary_data.csv")

    with open(summary_path, "w") as f:
        f.write("PageID\tMatrixIndex\tPageTitle\tSummary\n")
        for i in range(20):
            title = chr(65 + i) * 4
            f.write(f"{i*2}\t{i}\t{title}\t{title} {lormum_ipsum}\n")

    return summary_path


def test_read_summary(summary_data: str):
    with open(summary_data) as f:
        f.readline()

        assert f.readline().split("\t")[2] == "AAAA"
        assert int(f.readline().split("\t")[1]) == 1
        assert "CCCC " + lormum_ipsum == f.readline().split("\t")[3][:-1]


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


def test_read_matidx_info(matidx_info: str):
    matidx_df = pd.read_csv(matidx_info, sep="\t", header=0)

    assert np.all(np.arange(20) == matidx_df["MatrixIndex"].values)
    assert matidx_df["PageTitle"].values[2] == "CCCC"
    assert isinstance(matidx_df["Offset"].values[0], np.int64)


def test_AgentToolsDisk(hdf5_file: str, matidx_info: str, summary_data: str):
    # Create an instance of AgentToolsDisk using the provided file paths
    with AgentToolsDisk(hdf5_file, matidx_info, summary_data) as at:
        # Test the get_article_id method
        assert at.get_article_id("EEEE") == 4

        # Test the get_forward_links method
        assert np.all(at.get_forward_links(1) == np.arange(5, 10, 1))

        # Test the get_article_title method
        assert at.get_article_title(1) == "BBBB"

        # Test the get_article_summary method
        assert at.get_article_summary(5) == "FFFF " + lormum_ipsum

        # Test the get_forward_summaries method
        summary_list = at.get_forward_summaries(5)
        assert isinstance(summary_list, list)
        assert len(summary_list) == 5
        assert summary_list[0][0] == 10
        assert summary_list[0][1][:10] == "KKKK Lorem"

        # Test the get_forward_titles method
        title_list = at.get_forward_titles(5)
        assert isinstance(title_list, list)


def test_AgentToolsRAM(hdf5_file: str, summary_data: str):
    # Create an instance of AgentToolsRAM using the provided file paths
    at = AgentToolsRAM(hdf5_file, summary_data)

    # Test the get_article_id method
    assert at.get_article_id("EEEE") == 4

    # Test the get_forward_links method
    assert np.all(at.get_forward_links(1) == np.arange(5, 10, 1))

    # Test the get_article_title method
    assert at.get_article_title(1) == "BBBB"

    # Test the get_article_summary method
    assert at.get_article_summary(5) == "FFFF " + lormum_ipsum

    # Test the get_forward_summaries method
    summary_list = at.get_forward_summaries(5)
    assert isinstance(summary_list, list)
    assert len(summary_list) == 5
    assert summary_list[0][0] == 10
    assert summary_list[0][1][:10] == "KKKK Lorem"

    # Test the get_forward_titles method
    title_list = at.get_forward_titles(5)
    assert isinstance(title_list, list)
