# Wiki Game Agent `agents`

The agents in this package will play the Wikipedia game. There is an abstract class of `Agent`, which the derived classes will inherit from.

## Data
This package is currently only designed to be implemented with Wikipedia data, which is not saved to the repo due to size. A class called `AgentTools` has been designed to load and manage the data. The data contain unique IDs for each page in the Wikipedia data dump and through an instance of `AgentTools` it is possible to get the title, summary and links on any page given its unique ID.

## Agent Classes

Each dervied class will be initialised by passing an instance of `AgentTools` and also the maximum number of rounds that can be played in each game. A game is when an agent is given one pair of start and end (or source and target) pages and it has to navigate from the start to the end. If it fails to do so within the maximum number of steps (or rounds), then the game is logged as a failure.

### GreedyEmbeddingAgent
This implementation uses a pre-trained sentence transformer model to play the Wikipedia game. It uses the model to encode the summaries of all links on a page, and selects the one that is most similar to the encoded summary of the target page. This agent is not allowed to navigate back to a page it has already visited, as this approach will cause to enter a loop; therefore, this agent may also fail to complete a game if it reaches a page where there are no links available that it has not already visited.

The following shows an example of how this agent can be used to play a single game.

```
import os
from agents.agent_models.agent import GreedyEmbeddingAgent
from agents.agent_models.agent_tools import AgentTools
from agents.utils.paths import get_data_dir

data_dir = get_data_dir(__file__)

hdf5_file_path = os.path.join(data_dir, "wiki_forward.hdf5")
matrix_info_path = os.path.join(data_dir, "matidx_sum_table.csv")
summary_path = os.path.join(data_dir, "summary.csv")

tools = AgentTools(hdf5_file_path, matrix_info_path, summary_path)

agent = GreedyEmbeddingAgent(tools, max_rounds=20, model_name="all-distilroberta-v1")
agent.play_game("Ofotbanen (company)", "Raptorial", verbose=True)
agent.save_games("temp", "all-distilroberta-v1")

```

Alternatively, an agent can play multiple games by passing a list of source and target IDs through to `play_games` as follows:
```
agent.play_games(source_ids.tolist(), target_ids.tolist(), "output_folder_name", "output_file_name")
```
The `play_games` method will save the output of the games at regular intervals, where as the singular `play_game` method will not save the output by default.

### LLMPromptAgent
Not yet implemented, but this will use Azure's OpenAI service to play the game with a GPT engine.
