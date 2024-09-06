from __future__ import annotations

import os

# This file contains paths relative to the project root that can be imported.
# The 'find_project_root' function might need to be updated if the package is
# run on a location other than Baskerville or in the repo

BASK_ROOT_DIR = "/bask/projects/v/vjgo8416-wikigame/ARC-Wiki-Game"


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
    while (
        os.path.split(temp_dir)[1] != "agents"
        or os.path.split(os.path.dirname(temp_dir))[1] == "site-packages"
        or os.path.split(os.path.dirname(temp_dir))[1] == "src"
    ) and len(temp_dir) > 1:
        if os.path.split(temp_dir)[1] == "bask":
            return os.path.join(BASK_ROOT_DIR, "agents")
        temp_dir = os.path.dirname(temp_dir)

    if len(temp_dir) > 1:
        return temp_dir
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
    Given a file path, returns the `output_data` directory

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
    Given a file path, returns the `input_data` directory

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


def get_game_dir(data_path: str):
    """
    Given a path, returns the `game_data` directory

    Parameters
    ----------
    data_path: str
        The path of the data folder

    Returns
    -------
    str
        The file path of the output directory
    """
    return os.path.join(data_path, "game_data")


def get_wiki_data_paths(data_path: str):
    """
    Given a file path, returns the `output` directory

    Parameters
    ----------
    data_path: str
        The path of the data folder

    Returns
    -------
    str
        The file path of the three wikipedia data files
    """

    data_dir = get_input_dir(data_path)

    hdf5_file_path = os.path.join(data_dir, "wikipedia", "wiki_forward.hdf5")
    matrix_info_path = os.path.join(data_dir, "wikipedia", "matidx_sum_table.csv")
    summary_path = os.path.join(data_dir, "wikipedia", "summary.csv")

    return hdf5_file_path, matrix_info_path, summary_path
