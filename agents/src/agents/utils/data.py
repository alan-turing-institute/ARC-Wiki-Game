from __future__ import annotations

import os

from agents.agent_models.tools import AgentToolsBase, AgentToolsDisk, AgentToolsRAM


def get_data_folders(data_path: str) -> tuple[str, str, str]:
    """
    Checks that the game, input and output paths exist for the
    given parent data path. If they do exists then the function
    returns the three paths.

    Parameters
    ----------
    data_path: str
        Parent path of the data folders

    Returns
    -------
    game_data: str
        The path of the game data folder
    input_data: str
        The path of the input data folder
    output_data: str
        The path of the output data folder

    """
    game_data = os.path.join(data_path, "game_data")
    input_data = os.path.join(data_path, "input_data")
    output_data = os.path.join(data_path, "output_data")

    assert os.path.exists(game_data), "Path for game data does not exist: " + game_data
    assert os.path.exists(input_data), (
        "Path for input data does not exist: " + input_data
    )
    assert os.path.exists(output_data), (
        "Path for output data does not exist: " + output_data
    )
    return game_data, input_data, output_data


def get_input_data_paths(data_path: str, dataset_name: str) -> tuple[str, str, str]:
    """
    Parameters
    ----------
    data_path: str
        Parent path of the data folders
    dataset_name: str
        The name of the input dataset
    """
    _, input_path, _ = get_data_folders(data_path)
    return (
        os.path.join(input_path, dataset_name, dataset_name + "_forward_links.hdf5"),
        os.path.join(input_path, dataset_name, dataset_name + "_matrix_index_info.tsv"),
        os.path.join(input_path, dataset_name, dataset_name + "_summary.tsv"),
    )


def load_tools(
    data_path: str,
    dataset_name: str,
    load_data_to_ram: bool,
    old_title_format: bool = False,
) -> AgentToolsBase:
    """
    Parameters
    ----------
    data_path: str
        Parent path of the data folders
    dataset_name: str
        The name of the input dataset
    load_data_to_ram: bool
        True if data is to be loaded to RAM (might not be feasible if running
        locally) and False if the data are to remain on the hard-drive and
        loaded dynamically during run time.
    old_title_format: bool
        True if the data contain the original title format, False otherwise

    Returns
    -------
    AgentToolsBase:
        An object of a derived class of AgentToolsBase
    """
    # If or when we introduce more datasets, we will need to make the input dataset
    # an argument that will determine which one is loaded.
    hdf5_file_path, matrix_info_path, summary_path = get_input_data_paths(
        data_path, dataset_name
    )

    if load_data_to_ram:
        return AgentToolsRAM(
            hdf5_file_path,
            summary_path,
            sort_data=True,
            old_title_format=old_title_format,
        )
    return AgentToolsDisk(
        hdf5_file_path,
        matrix_info_path,
        open(summary_path, "rb"),  # noqa: SIM115
        old_title_format=old_title_format,
    )
