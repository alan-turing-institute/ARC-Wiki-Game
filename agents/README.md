# Agents

The agents in this package will play the Wikipedia game.

## Table of Contents
- [Getting Started](#getting-started)
- [Input Data: Graph Datasets](#input-data-graph-datasets)
- [Agent Classes](#agent-classes)
- [Playing a Single Game](#playing-a-single-game)
- [Playing Multiple Games](#playing-multiple-games)

## Getting Started
Assuming the repo has already been cloned, change directory to the `agents` package and install with `pip`:

   ```bash
   pip install .
   ```

Graph datasets in the required format to be used by the agents to play the Wikipedia game can be created using the [graph-prep package](../graph-prep/). The files for each graph dataset should be stored in sub-folders in the [data/input_data](data/input_data/) folder, with one sub-folder for each dataset.

A file containing a list of multiple source and target articles representing multiple games can also be generated using the [graph-prep package](../graph-prep/). These files should be stored in sub-folders in the [data/game_data](data/game_data/) folder, with one sub-folder for each graph datase in [data/input_data](data/input_data/). The names of the sub-folders in [data/input_data](data/input_data/) and [data/game_data](data/game_data/) should match.

If OpenAI models are to be used in a `PromptAgent` object (see [PromtAgent](#promptagent)), then the following environment variables must be set:
```
export AZURE_OPENAI_API_KEY="REPLACE_WITH_YOUR_KEY_VALUE_HERE"
export AZURE_OPENAI_ENDPOINT="REPLACE_WITH_YOUR_ENDPOINT_HERE"
```

If Ollama models are to be used in a `PromptAgent` object, then [Ollama](https://ollama.com/) must be installed.

## Input Data: Graph Datasets
This package is designed to be implemented with a graph dataset stored in one of two specific formats, these can be created using the [graph-prep package](../graph-prep/), which provides more details on that process.

### Format of Graph Datasets

There are two possible formats for the graph datasets. The first format expects the following files, where `dataset_name` is replaced with the given name of the graph dataset:
* `dataset_name_summary.tsv` - This file holds the summaries of the articles in a tab-separated file. It also has the PageID, MatrixIndex, and PageTitle before the Summary field.
* `dataset_name_title_info.tsv` - This file holds the information of all the articles and redirects in the wiki. It includes PageID, MatrixIndex, PageTitle, IsRedirect, and RedirectTitle. These are all ordered by the PageTitle for easier searching.
* `dataset_name_matrix_index_info.tsv` - This file holds the info on the articles ordered by the Matrix Index. It holds the PageID, MatrixIndex, PageTitle, and Offset. The Offset refers to the number of bytes the relevant page is in the summary file.
* `dataset_name-forward.hdf5` - This has the graph representation for the dataset in the form of a lookup for each page with what articles it links to.

The second format stores the article titles and summaries in a [DuckDB Database](https://duckdb.org/) and expects the following files, where `dataset_name` is replaced with the given name of the graph dataset:
* `datase_name_info.ddb` - To do description.
* `dataset_name_forward_links.hdf5` - This has the graph representation for the dataset in the form of a lookup for each page with what articles it links to.

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
    data_path = '/my-path/arc-wiki-game/ARC-Wiki-Game/agents/data'
    input_dataset_name = 'en_wiki'
    load_data_to_ram = True
    use_old_title = False
    from_database = False

    tools = load_tools(
        data_path, input_dataset_name, load_data_to_ram, use_old_title, from_database
    )
```

## Agent Classes

There is an abstract class of `Agent`, located in [src/agents/agent_models/agent.py](src/agents/agent_models/agent.py), from which the derived classes inherit. An instance of an `Agent` class will be able to play the Wikipedia game on a given dataset. A game is when an agent is given one pair of start and end (or source and target) articles and it has to navigate from the start to the end. If it fails to do so within the maximum number of steps (or rounds), then the game is logged as a failure.

Each dervied class will be initialised with two common arguments:
- `agent_tools` - an instance of an `AgentTools` class (see [Agent Tools](#agent-tools-classes))
- `max_rounds` - the maximum number of rounds that can be played in each game.

Each derived class will have the following methods, with varying implementation:
- `play_game` - this method will take the source and target article IDs or titles and will implement an approach to find a route between the two articles.
- `play_games` - this method plays multiple games, given a list of source and target article IDs, there is also an option to save the output every `n` games
- `record_game` - this method saves the details and outcome of a completed game to a list of games played, which is stored in the object
- `save_games` - this method saves the list of games played to a tab separated file, the path and name of which are passed as input
- `get_games_played` - this method returns the list of games played by the agent object

The implentation does not allow an to navigate back to a page it has already visited while it is playing the game. An agent may fail to complete a game if it does not reach the target page within the maximum number of rounds (steps) allowed and also if it reaches a page where there are no links available that it has not already visited. Some classes may have additional failure reasons.

A shot description of the classes that derive from `Agent` will be proivded and also any other initialising arguments or failure reasons that are unique to them.

### TFIDFAgent
This implementation plays the game using a [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) model, the corpus for the TF-IDF model can be created using the same dataset used to play the game, or there is an option for a different dataset to be used as the corpus. The algorithm selects the link that has a summary or title that is scored as being most similar to the target page summary using the cosine similarity.

Additional arguments that are passed when an instance of this class is created are:
- `lang_iso_code` - the iso code for the language to be used to create the corpus, this will influence which words are excluded from the dictionary
- `manual_seed` - the seed to use to select articles to use in the corpus
- `max_articles` - the maximum number of articles to use to create the corpus
- `agent_tools_for_corpus` - if `None` then the `agent_tools` existing object is used to create the corpus, otherwise a different `agent_tools` object can be passed using this argument

### GreedyEmbeddingAgent
This implementation uses a pre-trained sentence transformer model to play a game. It uses the model to encode the titles or summaries of all links in an article, and selects the one that is most similar to the encoded summary of the target article.

Arguments that are specific to use this agent to play a game are:
- `model_name` - this has to match the name of an existing model in the [sentence-transformer HuggingFace library](https://huggingface.co/models?library=sentence-transformers)
- `use_titles` - if `True` then this will encode the titles of each linked article, and if `False` it will encode the summaries


### PromptAgent
This agent uses a generative LLM to play the game. This plays the game by providing a prompt to the language model telling it is helping to play the Wikipedia game and the target article it needs to navigate towards. A list of titles of linked articles from the current article will then be provided to the language model and its job is to select one.

This class takes the following additional arguments for initialisation:
- `model` - an object of type `GenLM`, which is a class containing a language model and will be described below
- `num_retries` - the number of times the agent will retry to select a title if the language model hallucinates by generating title selected is not in the list of titles provided or failing to provide the output in the required JSON format
- `max_titles` - the maximum number of titles provided to the model

`GenLM` is an abstract class for holding a language model. It contains one method `get_response` which returns the response of the language model given a list of messages. There are two derived classes of `GenLM`:

- `OllamaModel` - this model calls the Ollama API to get responses
- `OpenAIModel` - this allows an OpenAI model that has been deployed using the Azure OpenAI Service to be used

## Playing a Single Game

The script [play_single_game.py](scripts/play_single_game.py) can be run to play an instance of the Wikipedia game. The possible arguments to use are all documented in the script. An example of how to run the script to navigate on a graph dataset of English Wikipedia from `Ofotbanen (company)` to `Raptorial`:

```
python play_single_game.py -p=../data/ --input_dataset=en_wiki --agent=GreedyEmbeddingAgent --model=intfloat/multilingual-e5-small --source=Ofotbanen\ \(company\) --target=Raptorial --verbose=True --load_data_to_ram=False
```

This script is only designed to be used for ad-hoc testing.

## Playing Multiple Games

The script [play_games.py](scripts/play_games.py) sets up an agent to play a pre-determined list of games (defined by source and target IDs). This script requires two additional files to set up the games: one file to store a list of the games to be played and another to record the configuration for models to play the list of games.

### List of Games
At least one file of multiple games should be stored in a sub-folder in [data/game_data](data/game_data/). There should be one sub-folder per graph dataset in the [data/input_data](data/input_data/) folder.

Each game file contains pairs of source and target article IDs - each pair representing a single game. These files can be generated in the [graph-prep package](../graph-prep/). When this is generated from the [graph-prep package](../graph-prep/), it can also include the optimal number of steps (the shortest possible path) between the two articles, which will be recorded in the output.

### Configuration

In each dataset subfolder of [data/game_data](data/game_data/) should be a `config.yaml` file. The `config.yaml` file will contain the details of the experiments that will be run across the dataset pairs. The structure of the config file is:

```
experiments:
  initial_test_tfidf:
    agent_class: TFIDFAgent
    game_datasets:
      - wikipedia_test_100
    models:
      - tfidf
    params:
      max_rounds: 100
      lang: en
      manual_seed: 42
      max_articles: 100
    agent_tools_for_corpus: another_dataset_name
    params_for_corpus:
      from_database: True

  initial_test_embedding:
    agent_class: GreedyEmbeddingAgent
    game_datasets:
      - wikipedia_test_100
    models:
      - all-distilroberta-v1
      - average_word_embeddings_glove.6B.300d
      - all-MiniLM-L6-v2
    params:
      max_rounds: 100

  initial_test_ollama:
    agent_class: PromptAgent
    game_datasets:
      - wikipedia_test_100
    models:
      - llama3.1:8b
      - gemma2:27b
    params:
      llm: ollama
      max_rounds: 100
      temperature: 0
      top_p: 0.1
      manual_seed: 42
      num_retries: 5
      max_titles: 200

  initial_test_openai:
    agent_class: PromptAgent
    game_datasets:
      - wikipedia_test_100
    models:
      - wikigame-gpt-4o-mini
    params:
      llm: gpt
      max_rounds: 100
      temperature: 0
      top_p: 0.1
      manual_seed: 42
      num_retries: 5
      max_titles: 500
```

Each experiment will be given a name, in this example the experiments are named `initial_test_tfidf`, `initial_test_embedding`, `initial_test_ollama` and `initial_test_openai`. The other arguments that need to be specified are:
- `agent_class` - the agent class to be used for the experiment: either `TFIDFAgent`, `GreedyEmbeddingAgent`, or `PromptAgent`, an example is given for all three above
- `game_datasets` - the dataset(s) to run the agent over, this name must match one of the datasets in the same folder as the config file
- `models` - a list of the models to be used in the agent. The only option for the `TFIDFAgent` is a model named `tfidf`, but the `GreedyEmbeddingAgent` and `PromptAgent` have multiple options, which must match the name of a sentence transformer model or a deployed prompting model, respectively.
- `params` - a list of parameters that are used to set characteristics of the model and game environment. One mandatory parameter is `max_rounds`, which determines the maximum number of rounds to be played (number of steps to be taken) before a game is recorded as a failure. Additional parameters can be set for loading the agent tools (such as `from_database`) and yet further parameters can be set for the `TFIDFAgent` and `PromptAgent`, which have some additional configuration.
- `agent_tools_for_corpus` - this is only an option for the `TFIDFAgent` and is not mandatory. If required, this is to be populated with the name of the dataset (which should exist in a subfolder of the given name in `data/input_data`) that is to be used to create the corpus.
- `params_for_corpus` - this is only required if the `agent_tools_for_corpus` is set and allows parameters to be set to load the agent tools for the additional dataset.

The experiments can be run by running the script `play_games.py` which takes arguments that are documented in the script. These argument will not change the outcome of the games that are run, but can be used to change how they are run. For example, a single experiment from a config file can be run, or all experiments can be run. Similarly, there is one argument which sets a limit on the number of games that can be run and another that controls how frequently the output is saved.

The script will create a sub-folder of [data/output_data](data/output_data/) with the name of the sub-folder matching the name of the input dataset and sub-folders will be created - one for each experiment that is run.
Each output file therefore resides in `data/output_data/<input_dataset_name>/<experiment_name>` has the naming convention:
```
<input_dataset_name>_<experiment_name>_<model_name>_<start_index>.csv
```

where `<model_name>` is the name of the model that was selected to run, e.g., `all-distilroberta-v1`, and `<start_index>` is an integer representing the index from the list of games where the agent started playing.
If an output file already exists for one combination of input dataset, experiment name, model and start index, then that combination will not be re-run (so no output files will get overwritten).
