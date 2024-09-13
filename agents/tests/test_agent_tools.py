from __future__ import annotations

import numpy as np

from agents.agent_models.tools import AgentToolsDisk, AgentToolsRAM


def test_AgentToolsDisk(
    hdf5_file: str, matidx_info: str, summary_data: str, lormum_ipsum: str
):
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


def test_AgentToolsRAM(hdf5_file: str, summary_data: str, lormum_ipsum: str):
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
