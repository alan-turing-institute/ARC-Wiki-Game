# Game Data

This folder stores datasets of pairs of source and target IDs that are used for testing over multiple agents. The data are stored in subfolders, with one subfolder per input dataset. Currently, there is only one input dataset and, therefore, only one subfolder: `wikipedia`.

The data can be generated using the script `generate_game_data.py`, this requires arguments specifying the number of records to generate (`num_records`), the name to call the generated dataset (`datasest_name`) and the name of the input dataset to use (`input_dataset`). Currently, the only valid name of the input dataset is `wikipedia`.

In each dataset subfolder there should be a `config.yaml` file and one or more dataset files of games. Each dataset file will contain a list of source (start) and target (end) page IDs - one pair representing each game. The `config.yaml` file will contain the details of the experiments that will be run across the dataset pairs. The structure of the config file is:

```
experiments:
  initial_test_100:
    agent_class: GreedyEmbeddingAgent
    game_datasets:
      - wikipedia_test_100
    models:
      - all-distilroberta-v1
      - average_word_embeddings_glove.6B.300d
      - all-MiniLM-L6-v2
      - paraphrase-albert-small-v2
      - all-mpnet-base-v2
      - all-roberta-large-v1
    params:
      max_rounds: 50
```

Each experiment will be given a name, in this example `initial_test_100`. The other arguments that need to be specified are:
- `agent_class` - the agent class to be used for the experiment (currently the only one available is `GreedyEmbeddingAgent`)
- `game_datasets` - the dataset to run the agent over, this name must match one of the datasets in the same folder as the config file
- `models` - a list of the models to be used in the agent. In this example, these are embedding models but for other agents they may be different types of model, e.g., 'GPT-3' or 'GPT-4'.
- `params: max_rounds` - the maximum number of rounds each game should be played for.

The experiments can be run by running the script `play_games.py` which only takes one argement which is the name of the input dataset (`input_dataset`). This will then load the config file associated with that dataset and run all the experiments listed in the dataset. If an output file already exists for one combination of game dataset and model, then that combination will not be re-run (so no output files will get overwritten). The `play_games.py` script also has optional arguments so that a single experiment and / or model can be passed to the script and only these combination(s) will be run. 
