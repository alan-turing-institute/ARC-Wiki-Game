from __future__ import annotations

import csv
from io import BufferedReader
from warnings import warn

import h5py
import numpy as np
import pandas as pd


class AgentToolsBase:
    def __init__(
        self,
        hdf5_file_path: str,
        info_dataframe: pd.DataFrame,
        old_title_format: bool = False,
    ):
        """Base class for AgentTools.

        AgentTools is a class that provides the accessible ids, titles, and summaries.

        Args:
            hdf5_file_path (str): The file path to the HDF5 file. This HDF5 file
                contains the forward links sorted in the order of the Matrix Index.
                It also contains a lookup table that maps the Matrix Index to the
                forward link locations.
            info_dataframe (pd.DataFrame): The DataFrame containing the Matrix Index
                and the title of the article.
            old_title_format (bool, optional): If the title is stored under the "Title"
                column as opposed to "PageTitle". Defaults to False.

        """
        print("Load the link data")
        with h5py.File(hdf5_file_path, "r") as h5_file:
            dset = h5_file["forward_vals"]
            self.forward_vals = dset[:]
            dset = h5_file["forward_locs"]
            self.forward_locs = dset[:]

        print(f"Loaded {len(self.forward_vals):_d} forward links")

        if np.all(np.diff(info_dataframe["MatrixIndex"].values) >= 0):
            print("Matrix Index is sorted")
        else:
            msg = "Matrix Index is not sorted"
            raise ValueError(msg)

        title_format = "Title" if old_title_format else "PageTitle"
        self.title_ary = info_dataframe[title_format].values
        self.matidx_ary = info_dataframe["MatrixIndex"].values

    def get_forward_links(self, article_id: int) -> np.ndarray:
        """Get the reachable links for the given article ID.

        Args:
            article_id (int): The article ID for which the forward links are to be
                fetched.

        Returns:
            np.ndarray: The forward links for the given article ID as a numpy array.
        """
        return self.forward_vals[
            self.forward_locs[article_id, 0] : self.forward_locs[article_id, 1]
        ]

    def get_article_title(self, article_id: int) -> str | None:
        """Get the title of the article for the given article ID.

        Args:
            article_id (int): The article ID for which the title is to be fetched.

        Returns:
            str: The title of the article.
        """
        idx = np.searchsorted(self.matidx_ary, article_id)
        if idx < len(self.matidx_ary) and self.matidx_ary[idx] == article_id:
            return self.title_ary[idx]
        warn(
            f"Article ID {article_id} not found in the Matrix Index Info Table",
            stacklevel=1,
        )
        return None

    def get_forward_titles(self, article_id: int) -> list[list[int | str | None]]:
        """Get the titles of the articles linked to the given article ID.

        Args:
            article_id (int): The article ID for which the titles are to be fetched.

        Returns:
            List[List[int, str]]: A list of lists containing the article ID and the
                title of the articles linked to the given article ID.
        """
        forward_links = self.get_forward_links(article_id)
        titles = []
        for link in forward_links:
            titles.append([int(link), self.get_article_title(link)])
        return titles

    def get_article_id(self, title: str) -> int | None:
        """Get the article ID for the given article title.

        Args:
            title (str): The title of the article for which the article ID is to be
                fetched.

        Returns:
            int: The article ID for the given article title.
        """
        title_loc = np.where(self.title_ary == title)[0]
        if len(title_loc) == 0:
            warn(
                f"Title {title} not found in the Matrix Index Info Table", stacklevel=1
            )
            return None
        if len(title_loc) > 1:
            warn(
                f"Multiple entries found for Title {title} in the Matrix Index Info "
                "Table",
                stacklevel=1,
            )
            return None
        return self.matidx_ary[title_loc[0]]


class AgentToolsDisk(AgentToolsBase):
    def __init__(
        self,
        hdf5_file_path: str,
        matidx_info_path: str,
        summary_data: str | BufferedReader,
        old_title_format: bool = False,
    ):
        """Initialize the AgentTools object.

        AgentTools is a class that provides the accessible ids, titles, and summaries
        for the articles in the Wikipedia dump. It can be open as a context manager
        to ensure that the Summary file is closed properly. For example:
        `with AgentTools(hdf5_file_path, matidx_info_path, summary_data) as agent:`

        Args:
            hdf5_file_path (str): The file path to the HDF5 file. This HDF5 file
                contains the forward links sorted in the order of the Matrix Index.
                It also contains a lookup table that maps the Matrix Index to the
                forward link locations.
            matidx_info_path (str): The file path to the Matrix Index Info table. This
                table contains the Matrix Index, the offset in the Summary file, and
                the title of the article.
            summary_data (str, BufferedReader): The file with all the Summary data.
                This file contains the summaries of the articles extracted from the
                dump. The offset relates to the byte location of the summary in the
                file. This can be a file path or a file object. If it is a file can
                object, it should be opened in binary mode. `open(file_path, "rb")`.
            old_title_format (bool, optional): If the title is stored under the "Title"
                    column as opposed to "PageTitle". Defaults to False.

        Raises:
            ValueError: If the Matrix Index is not sorted.
        """
        print("Loading the Matrix Index Info Table")
        matidx_info_df = pd.read_csv(
            matidx_info_path,
            sep="\t",
            header=0,
            quoting=csv.QUOTE_NONE,
        )

        self.sum_offset_ary = matidx_info_df["Offset"].values

        super().__init__(hdf5_file_path, matidx_info_df, old_title_format)

        if isinstance(summary_data, str):
            self._summary_path = summary_data
            self.summary_file = None
        elif isinstance(summary_data, BufferedReader):
            self.summary_file = summary_data
        else:
            msg = (
                "Invalid type for the Summary needs to be a file path or a file "
                "object opened in reading binary mode."
            )
            raise TypeError(msg)

    def __enter__(self) -> AgentToolsDisk:
        """Initialize the AgentTools object."""
        print("Loading the Summary File")
        self.summary_file = open(self._summary_path, "rb")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the AgentTools object."""
        self.summary_file.close()

    def get_article_summary(self, article_id) -> str | None:
        """Get the summary of the article for the given article ID.

        Args:
            article_id (int): The article ID for which the summary is to be fetched.

        Returns:
            str: The summary of the article, from the Summary file.
        """
        if self.summary_file is None:
            msg = (
                "Summary file is not open, use with statement "
                "`with AgentToolsDisk(...)` to open the file or provide a file object."
            )
            raise ValueError(msg)

        idx = np.searchsorted(self.matidx_ary, article_id)
        if idx < len(self.matidx_ary) and self.matidx_ary[idx] == article_id:
            offset = self.sum_offset_ary[idx]
        else:
            warn(
                f"Article ID {article_id} not found in the Matrix Index Info Table",
                stacklevel=1,
            )
            return None

        self.summary_file.seek(offset, 0)
        return str(self.summary_file.readline(), encoding="utf-8").split("\t")[-1][:-1]

    def get_forward_summaries(self, article_id: int) -> list[list[int | str | None]]:
        """Get the summaries of the articles linked to the given article ID.

        Args:
            article_id (int): The article ID for which the summaries are to be fetched.

        Returns:
            List[List[int, str]]: A list of lists containing the article ID and the
                summary of the articles linked to the given article ID.
        """
        forward_links = self.get_forward_links(article_id)
        summaries = []
        for link in forward_links:
            summaries.append([int(link), self.get_article_summary(link)])
        return summaries


class AgentToolsRAM(AgentToolsBase):
    def __init__(
        self,
        hdf5_file_path: str,
        summary_data_path: str,
        old_title_format: bool = False,
        sort_data: bool = False,
    ):
        """Initialize the AgentTools object.

        AgentTools is a class that provides the accessible ids, titles, and summaries
        for the articles in the Wikipedia dump. It can be open as a context manager
        to ensure that the Summary file is closed properly. For example:
        `with AgentTools(hdf5_file_path, matidx_info_path, summary_data) as agent:`

        Args:
            hdf5_file_path (str): The file path to the HDF5 file. This HDF5 file
                contains the forward links sorted in the order of the Matrix Index.
                It also contains a lookup table that maps the Matrix Index to the
                forward link locations.
            summary_data_path (str): The file path to the Summary data table. This table
                contains the summaries of the articles extracted from the dump along
                with the Matrix Index and title.
            old_title_format (bool, optional): If the title is stored under the "Title"
                column as opposed to "PageTitle". Defaults to False.
            sort_data (bool, optional): If the Summary data table needs to be sorted.
                Defaults to False.

        Raises:
            ValueError: If the Matrix Index is not sorted.
        """
        print("Loading the Summary Data Table")
        summary_df = pd.read_csv(
            summary_data_path, sep="\t", header=0, quoting=csv.QUOTE_NONE
        )
        if sort_data:
            print("Sorting the Summary Data Table")
            summary_df.sort_values("MatrixIndex", inplace=True)
            summary_df.reset_index(drop=True, inplace=True)

        self.summary_ser = summary_df["Summary"]

        super().__init__(hdf5_file_path, summary_df, old_title_format=old_title_format)

    def get_article_summary(self, article_id) -> str | None:
        """Get the summary of the article for the given article ID.

        Args:
            article_id (int): The article ID for which the summary is to be fetched.

        Returns:
            str: The summary of the article, from the Summary file.
        """
        idx = np.searchsorted(self.matidx_ary, article_id)
        if idx < len(self.matidx_ary) and self.matidx_ary[idx] == article_id:
            summary = self.summary_ser.iloc[idx]
        else:
            warn(
                f"Article ID {article_id} not found in the Matrix Index Info Table",
                stacklevel=1,
            )
            return None

        return str(summary)

    def get_forward_summaries(self, article_id: int) -> list[list[int | str | None]]:
        """Get the summaries of the articles linked to the given article ID.

        Args:
            article_id (int): The article ID for which the summaries are to be fetched.

        Returns:
            List[List[int, str]]: A list of lists containing the article ID and the
                summary of the articles linked to the given article ID.
        """
        forward_links = self.get_forward_links(article_id)
        summaries = []
        for link in forward_links:
            summaries.append([int(link), self.get_article_summary(link)])
        return summaries


__all__ = ["AgentToolsBase", "AgentToolsDisk", "AgentToolsRAM"]
