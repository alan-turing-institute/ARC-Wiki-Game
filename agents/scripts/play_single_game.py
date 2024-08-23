# from __future__ import annotations

# import os
# import time

# from numpy import loadtxt
# from numpy.random import choice

# from agents.agent_models.agent import GreedyEmbeddingAgent
# from agents.agent_models.tools import AgentTools
# from agents.utils.paths import get_input_dir

# start_time = time.time()

# data_dir = get_input_dir(__file__)

# hdf5_file_path = os.path.join(data_dir, "wiki_forward.hdf5")
# matrix_info_path = os.path.join(data_dir, "matidx_sum_table.csv")
# summary_path = os.path.join(data_dir, "summary.csv")

# tools = AgentTools(hdf5_file_path, matrix_info_path, summary_path)


# end_time = time.time()
# interval = end_time - start_time
# print("Agent tools loaded. Time:%.2f" % (interval))

# start_time = time.time()

# temp_agent = GreedyEmbeddingAgent(
#     tools, max_rounds=50, model_name="all-distilroberta-v1"
# )
# temp_agent.play_game("Ofotbanen (company)", "Raptorial", verbose=True)
# temp_agent.save_games("temp", "all-distilroberta-v1")

# end_time = time.time()
# interval = end_time - start_time
# print("Time:%.2f" % (interval))
