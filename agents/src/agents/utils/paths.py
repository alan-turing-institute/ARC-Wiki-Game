from __future__ import annotations

import os

# This file contains paths relative to the project root that can be imported.
# If this file moves relative to the project root, find_repo_root() must be changed.


def find_project_root(file_path: str) -> str:
    """
    Returns root folder for the package `agents`

    Returns
    ----------
    str
        Path to the root `agents` directory.
    """

    temp_dir = os.path.dirname(file_path)
    while os.path.split(temp_dir)[1] != "agents" and len(temp_dir) > 0:
        temp_dir = os.path.dirname(temp_dir)

    if len(temp_dir) > 0:
        return temp_dir
    raise ValueError("Cannot find root directory for file path " + file_path)


def get_data_dir(file_path: str):
    return os.path.join(find_project_root(file_path), "data")


def get_output_dir(file_path: str):
    return os.path.join(find_project_root(file_path), "output")
