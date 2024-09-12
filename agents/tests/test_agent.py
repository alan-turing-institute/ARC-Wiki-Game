from __future__ import annotations

from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.agent_models.tools import AgentToolsDisk


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
