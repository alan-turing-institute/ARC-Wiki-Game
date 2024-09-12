from __future__ import annotations

import h5py
import numpy as np
import pandas as pd


def test_read_matidx_info(matidx_info: str):
    matidx_df = pd.read_csv(matidx_info, sep="\t", header=0)

    assert np.all(np.arange(20) == matidx_df["MatrixIndex"].values)
    assert matidx_df["PageTitle"].values[2] == "CCCC"
    assert isinstance(matidx_df["Offset"].values[0], np.int64)


def test_read_summary(summary_data: str, lormum_ipsum: str):
    with open(summary_data) as f:
        f.readline()

        assert f.readline().split("\t")[2] == "AAAA"
        assert int(f.readline().split("\t")[1]) == 1
        assert "CCCC " + lormum_ipsum == f.readline().split("\t")[3][:-1]


def test_read_hdf5(hdf5_file: str):
    with h5py.File(hdf5_file, "r") as h5_file:
        dset = h5_file["forward_vals"]
        forward_vals = dset[:]
        dset = h5_file["forward_locs"]
        forward_locs = dset[:]

        assert np.all(forward_vals[0:5] == np.arange(0, 5, 1))
        assert np.all(forward_locs[:, 0] == np.arange(0, 96, 5))
