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
    return os.path.join(find_project_root(file_path), "output")


def get_test_data_dir(file_path: str):
    """
    Given a file path, returns the `test_data` directory

    Parameters
    ----------
    file_path: str
        The file path from where the function is called

    Returns
    -------
    str
        The file path of the test data directory
    """
    return os.path.join(find_project_root(file_path), "test_data")
