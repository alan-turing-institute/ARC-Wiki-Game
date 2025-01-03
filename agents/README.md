# Wiki Game Agent

The agents in this package will play the Wikipedia game.

## Table of Contents
- [Input Data: Graph Datasets](#input-data-graph-datasets)


## Input Data: Graph Datasets
This package is designed to be implemented with a graph dataset. Graph datasets in the required format can be created using the [graph-prep package](../graph-prep/). The files for each graph dataset can be stored in sub-folders in the [data/input_data](data/input_data/) folder, with one sub-folder for each dataset - although it is feasible to load them from other locations as well (see [Loading Graph Datasets](#loading-graph-datasets))

### Agent Tools Classes
A faimily of classes found in [src/agents/agent_models/tools.py](src/agents/agent_models/tools.py) have been designed to load and manage the data. The `AgentToolsBase` is an abstract base class, which the other classes inherit from. The choice of class to use to load and manage the data will depend on the size of the dataset, the hardware being used to run the code and the storage format of the data:
- `AgentToolsDisk`
- `AgentToolsRAM`
- `AgentToolsDatabaseDisk`
- `AgentToolsDatabaseRAM`

The `Disk` or `RAM` options for the class determine whether the data is kept on the disk or loaded to RAM, respectively. Loading the data to RAM might not be feasible if the dataset is large, but could result in faster execution of code for some hardware. The `Database` option for the class is to be used when the article titles and summaries are stored in a database rather than files. The specific type of database expected in [DuckDB](https://duckdb.org/) and any other database type would require additional development.

The data should contain unique IDs for each article in the dataset and through an instance of `AgentTools` it is possible to get the title, summary and links on any page given its unique ID. All `AgentTools`, but with some varying implementations:
- `get_article_title` - takes an article ID as input and returns the title of the article
- `get_article_summary` - takes an article ID as input and returns the summary of the article
- `get_forward_links` - takes an article ID as input and returns the IDs of the forward (outward) links for the given article
- `get_forward_titles` - takes an article ID as input and returns the titles of the forward (outward) links for the given article
- `get_forward_summaries` - takes an article ID as input and returns the summaries of the forward (outward) links for the given article
- `get_article_id` - takes an article title as input and returns the ID of the given article only if the title is unique - otherwise `None` is returned with a warning

### Loading Graph Datasets
A graph dataset can be loaded to an instance of `AgentTools` by calling the `load_tools` function which is located in [src/agents/utils/data.py](src/agents/utils/data.py). The function takes the following arguments:
- `data_path` - the parent path of the data folders
- `dataset_name` - the name of the input dataset
- `load_data_to_ram` - `True` if data is to be loaded to RAM (might not be feasible if running locally) and `False` if the data are to remain on the hard-drive and loaded dynamically during run time.
- `old_title_format`- `True` if the data contain the original title format, `False` otherwise - unless using legacy code, this should always be set to `False`
- `from_database` - `True` if the data are stored in a DuckDB database

The following code example will load a dataset named `en_wiki` from some given loaction.

```
    data_path = /my-path/arc-wiki-game/ARC-Wiki-Game/agents/data
    input_dataset_name = en_wiki
    load_data_to_ram = True
    use_old_title = False
    from_database = False

    tools = load_tools(
        data_path, input_dataset_name, load_data_to_ram, use_old_title, from_database
    )
```

## Agent Classes

There is an abstract class of `Agent`, which the derived classes will inherit from.

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
