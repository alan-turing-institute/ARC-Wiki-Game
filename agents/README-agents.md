# Wiki Game Agent

The agents in this package will play the Wikipedia game. There is an abstract class of `Agent`, which the derived classes will inherit from.

## Data
This package is currently only designed to be implemented with Wikipedia data, which is not saved to the repo due to size. A class called `AgentTools` has been designed to load and manage the data. The data contain unique IDs for each page in the Wikipedia data dump and through an instance of `AgentTools` it is possible to get the title, summary and links on any page given its unique ID.

## Agent Classes

Each dervied class will be initialised by passing an instance of `AgentTools` and also the maximum number of rounds that can be played in each game. A game is when an agent is given one pair of start and end (or source and target) pages and it has to navigate from the start to the end. If it fails to do so within the maximum number of steps (or rounds), then the game is logged as a failure.

The `play_games` method will save the output of the games at regular intervals, where as the singular `play_game` method will not save the output, but this can be achieved by running the `save_game` method.

The script `play_single_game.py` can be run to play an instance of the Wikipedia game. Both `GreedyEmbeddingAgent` objects and `PromptAgent` objects can be used in this script.

A script `play_games.py` sets up an agent to play a pre-determined list of games (defined by source and target IDs). Only the `GreedyEmbeddingAgent` is currently able to play multiple games. 

### GreedyEmbeddingAgent
This implementation uses a pre-trained sentence transformer model to play the Wikipedia game. It uses the model to encode the summaries of all links on a page, and selects the one that is most similar to the encoded summary of the target page. This agent is not allowed to navigate back to a page it has already visited, as this approach will cause to enter a loop; therefore, this agent may also fail to complete a game if it reaches a page where there are no links available that it has not already visited.

Arguments that are specific to use this agent to play a game are the `model_name`. This has to match the name of an existing model in the 'sentence-transformer' HuggingFace library. Currently this is limited to the original models in the library (those authored by 'sentence-transformers'), but this restriction could be lifted if required in future.


### PromptAgent
This agent uses a generative LLM to play the game. 

Currently the LLM must be deployed using Azure's OpenAI service; however, this might be expanded in future to include other models. The following environment variables must be set:
```
export AZURE_OPENAI_API_KEY="REPLACE_WITH_YOUR_KEY_VALUE_HERE"
export AZURE_OPENAI_ENDPOINT="REPLACE_WITH_YOUR_ENDPOINT_HERE"

```

Currently this agent can only be run over a single game using the script `play_single_game.py`. 

Arguments that are specific to use this agent to play a game are:
- `model_name`: this must match the name of a deployed model for the endpoint stored in the environment for Azure's OpenAI service
- `temperature`: this is the temperature argument for GPT, a lower value makes the respones more deterministic
