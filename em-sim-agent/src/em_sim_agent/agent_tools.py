import h5py
import numpy as np
import pandas as pd


class AgentTools:
    def __init__(self, hdf5_file_path: str, matidx_info_path: str, summary_path: str):
        """Initialize the AgentTools object.

        Args:
            hdf5_file_path (str): The file path to the HDF5 file. This HDF5 file
                contains the forward links sorted in the order of the Matrix Index.
                It also contains a lookup table that maps the Matrix Index to the
                forward link locations.
            matidx_info_path (str): The file path to the Matrix Index Info table. This
                table contains the Matrix Index, the offset in the Summary file, and
                the title of the article.
            summary_path (str): The file path to the Summary file. This file contains
                the summaries of the articles extracted from the dump. The offset
                relates to the byte location of the summary in the file.

        Raises:
            ValueError: If the Matrix Index is not sorted.
        """
        print("Load the link data")
        with h5py.File(hdf5_file_path, "r") as h5_file:
            dset = h5_file["forward_vals"]
            self.forward_vals = dset[:]
            dset = h5_file["forward_locs"]
            self.forward_locs = dset[:]

        print(f"Loaded {len(self.forward_vals):_d} forward links")
        print("Loading the Matrix Index Info Table")
        matidx_info_df = pd.read_csv(matidx_info_path, sep="\t", header=0)

        if np.all(
            np.sort(matidx_info_df["MatrixIndex"].values)
            == matidx_info_df["MatrixIndex"].values
        ):
            print("Matrix Index is sorted")
        else:
            raise ValueError("Matrix Index is not sorted")

        self.matidx_ary = matidx_info_df["MatrixIndex"].values
        self.sum_offset_ary = matidx_info_df["Offset"].values
        self.title_ary = matidx_info_df["Title"].values

        print("Loading the Summary File")
        self.summary_file = open(summary_path, "rb")

    def get_forward_links(self, article_id: int):
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

    def get_article_title(self, article_id: int):
        """Get the title of the article for the given article ID.

        Args:
            article_id (int): The article ID for which the title is to be fetched.

        Returns:
            str: The title of the article.
        """
        idx = np.searchsorted(self.matidx_ary, article_id)
        if idx < len(self.matidx_ary) and self.matidx_ary[idx] == article_id:
            return self.title_ary[idx]
        else:
            return None

    def get_article_summaray(self, article_id):
        """Get the summary of the article for the given article ID.

        Args:
            article_id (int): The article ID for which the summary is to be fetched.

        Returns:
            str: The summary of the article, from the Summary file.
        """
        idx = np.searchsorted(self.matidx_ary, article_id)
        if idx < len(self.matidx_ary) and self.matidx_ary[idx] == article_id:
            offset = self.sum_offset_ary[idx]
        else:
            return None

        self.summary_file.seek(offset, 0)
        return str(self.summary_file.readline(), encoding="utf-8").split("\t")[-1]

    def get_forward_titles(self, article_id: int):
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

    def get_forward_summaries(self, article_id: int):
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
            summaries.append([int(link), self.get_article_summaray(link)])
        return summaries
