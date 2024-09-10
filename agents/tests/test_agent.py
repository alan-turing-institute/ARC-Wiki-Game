from __future__ import annotations

import h5py
import numpy as np
import pytest

from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.agent_models.tools import AgentToolsDisk

lormum_ipsum = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam vitae "
    "sagittis est. Aliquam ultrices, dui at molestie condimentum, elit turpis volutpat "
    "sapien, ac ultricies metus ipsum sit amet leo. Integer nulla nisl, aliquet "
    "convallis sagittis vel, varius at metus. Morbi eget lacus a dolor dapibus "
    "tincidunt eget eu quam. Maecenas eros mauris, pretium sit amet convallis nec, "
    "maximus ac tortor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Orci "
    "varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. "
    "Aliquam sed mauris pharetra, vehicula magna a, sodales sem. Nunc tincidunt augue "
    "sed ornare posuere. Suspendisse auctor augue at tortor condimentum venenatis. "
    "Curabitur fringilla efficitur dui, non volutpat diam maximus a. Nunc rhoncus "
    "lacinia mattis. Aliquam scelerisque ipsum a odio interdum aliquet ac ut felis."
)


@pytest.fixture(scope="module")
def testing_dir(tmp_path_factory: pytest.fixture.tmp_path_factory) -> str:
    # Create a temporary directory
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def hdf5_file(testing_dir: str) -> str:
    hdf5_path = str(testing_dir / "test_data.hdf5")

    # Create a graph to save to file
    fives = np.arange(0, 101, 5)
    test_locs = np.array([fives[0:-1], fives[1:]]).T
    twenty = np.arange(0, 20, 1)
    test_vals = np.concatenate(
        [
            twenty,  # One step
            twenty[::2],  # Two steps
            twenty[1::2],
            twenty[::4],
            twenty[1::4],  # Four steps
            twenty[2::4],
            twenty[3::4],
            twenty[::5],  # Five steps
            twenty[1::5],
            twenty[2::5],
            twenty[3::5],
            twenty[4::5],
            twenty[2::],  # Offset by 2
            np.array([0, 1]),
        ]
    )

    # Create and save some data to the HDF5 file
    with h5py.File(hdf5_path, "w") as f:
        # Create a dataset in the file
        f.create_dataset("forward_vals", data=test_vals)
        f.create_dataset("forward_locs", data=test_locs)

    return hdf5_path


@pytest.fixture(scope="module")
def summary_data(testing_dir: str) -> str:
    summary_path = str(testing_dir / "summary_data.csv")

    with open(summary_path, "w") as f:
        f.write("PageID\tMatrixIndex\tPageTitle\tSummary\n")
        for i in range(20):
            title = chr(65 + i) * 4
            f.write(f"{i*2}\t{i}\t{title}\t{title} {lormum_ipsum}\n")

    return summary_path


@pytest.fixture(scope="module")
def matidx_info(testing_dir: str, summary_data: str) -> str:
    offset_ary = np.zeros(20).astype(int)

    offset = 0  # Offset to keep track of the current position in the file in bytes
    with open(summary_data, "rb") as sum_f:
        offset += len(sum_f.readline())  # Skip the header
        for i in range(20):
            offset_ary[i] = offset
            offset += len(sum_f.readline())

    matidx_info_path = str(testing_dir / "matidx_info.csv")

    with open(matidx_info_path, "w") as f:
        f.write("PageID\tMatrixIndex\tOffset\tPageTitle\n")
        for i in range(20):
            f.write(f"{i*2}\t{i}\t{offset_ary[i]}\t{chr(65 + i) * 4}\n")

    return matidx_info_path


def test_GreedyEmbeddingAgent(
    hdf5_file: str, matidx_info: str, summary_data: str, testing_dir: str
):
    # Create an instance of AgentToolsDisk using the provided file paths
    with AgentToolsDisk(hdf5_file, matidx_info, summary_data) as at:
        agent_game = GreedyEmbeddingAgent(
            at,
            max_rounds=4,
            model_name="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        )
        agent_game.play_game("BBBB", "RRRR")
        agent_game.play_game("DDDD", 8)
        agent_game.play_game(8, "DDDD")
        agent_game.play_game(13, 5)
        games_played = agent_game.get_games_played()
        assert len(games_played) == 4
        assert games_played[0]["found_target"]
        assert games_played[0]["num_steps"] == 2
        assert len(games_played[0]["route"]) == 3
        assert not games_played[-1]["found_target"]
        assert games_played[-1]["num_steps"] == 4
        assert len(games_played[-1]["route"]) == 5

        agent_games = GreedyEmbeddingAgent(
            at,
            max_rounds=4,
            model_name="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        )
        source_ids = [1, 3, 8, 13]
        target_ids = [12, 8, 3, 5]
        agent_games.play_games(
            source_ids, target_ids, testing_dir, "temp_games", save_n_games=1
        )
        games_played = agent_game.get_games_played()
        successful_games = 0
        for game in games_played:
            if game["found_target"]:
                successful_games += 1
        assert successful_games == 2
        assert len(games_played) == 4
