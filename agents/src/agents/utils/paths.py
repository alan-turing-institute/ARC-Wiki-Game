from __future__ import annotations

import os

# This file contains paths relative to the project root that can be imported.
# If this file moves relative to the project root, find_repo_root() must be changed.


def find_project_root(file_path: str) -> str:
    """
    Returns root folder for the package `agents`

    Parameters
    ----------
    file_path: str
        The path of the file from which to find the root folder.

    Returns
    ----------
    str
        Path to the root `agents` directory.
    """

    temp_dir = os.path.dirname(file_path)
    while os.path.split(temp_dir)[1] != "ARC-Wiki-Game" and len(temp_dir) > 0:
        temp_dir = os.path.dirname(temp_dir)

    if len(temp_dir) > 0:
        return os.path.join(temp_dir, "agents")
    raise ValueError("Cannot find root directory for file path " + file_path)


def get_data_dir(file_path: str) -> str:
    """
    Given a file path, returns the `data` directory

    Parameters
    ----------
    file_path: str
        The file path from where the function is called

    Returns
    -------
    str
        The file path of the data directory
    """
    return os.path.join(find_project_root(file_path), "data")


def get_output_dir(file_path: str):
    """
    Given a file path, returns the `output` directory

    Parameters
    ----------
    file_path: str
        The file path from where the function is called

    Returns
    -------
    str
        The file path of the output directory
    """
    return os.path.join(get_data_dir(file_path), "output_data")


def get_input_dir(file_path: str):
    """
    Given a file path, returns the `output` directory

    Parameters
    ----------
    file_path: str
        The file path from where the function is called

    Returns
    -------
    str
        The file path of the output directory
    """
    return os.path.join(get_data_dir(file_path), "input_data")


def get_wiki_data_paths(file_path: str):
    """
    Given a file path, returns the `output` directory

    Parameters
    ----------
    file_path: str
        The file path from where the function is called

    Returns
    -------
    str
        The file path of the three wikipedia data files
    """

    data_dir = get_input_dir(file_path)

    hdf5_file_path = os.path.join(data_dir, "wiki_forward.hdf5")
    matrix_info_path = os.path.join(data_dir, "matidx_sum_table.csv")
    summary_path = os.path.join(data_dir, "summary.csv")

    return hdf5_file_path, matrix_info_path, summary_path
