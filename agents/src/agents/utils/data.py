from __future__ import annotations

from agents.agent_models.tools import AgentToolsBase, AgentToolsDisk, AgentToolsRAM
from agents.utils.paths import get_wiki_data_paths


def load_tools(load_data_to_ram: bool, old_title_format: bool) -> AgentToolsBase:
    """
    Parameters
    ----------
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
    hdf5_file_path, matrix_info_path, summary_path = get_wiki_data_paths(__file__)

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
