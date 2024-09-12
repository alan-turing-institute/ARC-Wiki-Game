from __future__ import annotations

from agents.agent_models.agent import GreedyEmbeddingAgent, PromptAgent
from agents.agent_models.generative_lm import MockGenLM
from agents.agent_models.tools import AgentToolsDisk


def test_GreedyEmbeddingAgent(
    hdf5_file: str, matidx_info: str, summary_data: str, testing_dir: str
):
    with AgentToolsDisk(hdf5_file, matidx_info, summary_data) as at:
        agent = GreedyEmbeddingAgent(
            at,
            max_rounds=4,
            model_name="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        )
        agent.play_game("BBBB", "RRRR")
        agent.play_game("DDDD", 8)
        agent.play_game(8, "DDDD")
        agent.play_game(13, 5)
        games_played = agent.get_games_played()
        assert len(games_played) == 4
        assert games_played[0]["found_target"]
        assert games_played[0]["num_steps"] == 2
        assert len(games_played[0]["route"]) == 3
        assert not games_played[-1]["found_target"]
        assert games_played[-1]["num_steps"] == 4
        assert len(games_played[-1]["route"]) == 5

        agent = GreedyEmbeddingAgent(
            at,
            max_rounds=4,
            model_name="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        )
        source_ids = [1, 3, 8, 13]
        target_ids = [12, 8, 3, 5]
        agent.play_games(
            source_ids, target_ids, testing_dir, "temp_games", save_n_games=1
        )
        games_played = agent.get_games_played()
        successful_games = 0
        for game in games_played:
            if game["found_target"]:
                successful_games += 1
        assert successful_games == 2
        assert len(games_played) == 4


def test_PromptAgent(hdf5_file: str, matidx_info: str, summary_data: str):
    with AgentToolsDisk(hdf5_file, matidx_info, summary_data) as at:
        responses_success = [
            {"type": "json", "link": "CCCC"},
            {"type": "json", "link": "LLLL"},
            {"type": "json", "link": "PPPP"},
        ]
        gen_model = MockGenLM(responses_success)
        agent = PromptAgent(at, max_rounds=5, model=gen_model)
        agent.play_game(0, 15)
        game = agent.get_games_played()[0]
        assert game["found_target"]
        assert game["route"] == [0, 2, 11, 15]
        assert game["num_steps"] == 3
        assert game["error_message"] == ""

        responses_halucinate = [
            {"type": "json", "link": "CCCC"},
            {"type": "json", "link": "LLLL"},
            {"type": "json", "link": "8888"},
            {"type": "json", "link": "8888"},  # first retry
        ]
        gen_model = MockGenLM(responses_halucinate)
        agent = PromptAgent(at, max_rounds=5, model=gen_model)
        agent.play_game(0, 15)
        game = agent.get_games_played()[0]
        assert not game["found_target"]
        assert game["route"] == [0, 2, 11, -1]
        assert game["num_steps"] == 2
        assert game["error_message"] == "Hallucination - link not in list"

        responses_max_steps_reached = [
            {"type": "json", "link": "CCCC"},
            {"type": "json", "link": "LLLL"},
            {"type": "json", "link": "HHHH"},
            {"type": "json", "link": "TTTT"},
            {"type": "json", "link": "AAAA"},
            {"type": "json", "link": "BBBB"},
        ]
        gen_model = MockGenLM(responses_max_steps_reached)
        agent = PromptAgent(at, max_rounds=5, model=gen_model)
        agent.play_game(0, 15)
        game = agent.get_games_played()[0]
        assert not game["found_target"]
        assert game["route"] == [0, 2, 11, 7, 19, 1]
        assert game["num_steps"] == 5
        assert game["error_message"] == "Maximum number of steps reached"

        responses_not_json = [
            {"type": "json", "link": "CCCC"},
            {"type": "json", "link": "LLLL"},
            {"type": "not_json"},
            {"type": "not_json"},  # first retry
        ]
        gen_model = MockGenLM(responses_not_json)
        agent = PromptAgent(at, max_rounds=5, model=gen_model)
        agent.play_game(0, 15)
        game = agent.get_games_played()[0]
        assert not game["found_target"]
        assert game["route"] == [0, 2, 11, -1]
        assert game["num_steps"] == 2
        assert game["error_message"] == "Response not in JSON format"
    assert True
