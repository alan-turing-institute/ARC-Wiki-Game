from __future__ import annotations

import argparse
import os

import yaml
from numpy import array, loadtxt, savetxt

from agents.agent_models.agent import (
    GreedyEmbeddingAgent,
    PromptAgent,
    ReasoningAgent,
    TFIDFAgent,
)
from agents.agent_models.generative_lm import OllamaLM, OpenAIModel
from agents.utils.data import get_data_folders, load_tools


def parse_args():
    parser = argparse.ArgumentParser(description="Get config file to run")

    parser.add_argument(
        "--input_dataset",
        required=True,
        help="The name of the input dataset which the experiments are to be run.\
             This is expected to match the name of a folder in the `input_data` folder",
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        required=False,
        help="The name of the experiment to run (optional). If not provided, all \
            experiments in the config file will run.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        help="The name of the model to run for the given experiment (optional). If not \
            provided, then the code will check to see if a model id has been provided. \
            If neither are provided then all models in the config file will run for \
            the given experiment.",
    )
    parser.add_argument(
        "--model_id",
        required=False,
        type=int,
        help="The id of the model to run for the given experiment (optional). This \
            will only be used if the model name has not been provided. \
            If neither are provided then all models in the config file will run for \
            the given experiment.",
    )
    parser.add_argument(
        "--load_data_to_ram",
        required=False,
        type=bool,
        default=True,
        help="True if data is to be loaded to RAM, False if it remains on hard drive",
    )
    parser.add_argument(
        "--save_n_games",
        required=False,
        type=int,
        default=100,
        help="An integer that represents how often the output will be saved",
    )
    parser.add_argument(
        "-p",
        "--data_path",
        required=True,
        help="The path that the data are stored in, e.g. \
            '/home/ARC-WIKI-GAME/agents/data/, this must include the 'data' folder.\
             The 'data' folder is assumed to have three subfolders: 'game_data',\
             'input_data' and 'output_data'",
    )
    parser.add_argument(
        "--max_games",
        required=False,
        default=5000,
        type=int,
        help="The maximum number of games that can be run for any combination of \
            agent, experiment and model. Might want to set to a low value, e.g. 10, \
            when testing locally.",
    )
    parser.add_argument(
        "--ollama_host",
        required=False,
        default=11434,
        help="The host port of the Ollama API",
    )
    return parser.parse_args()


def load_test_dataset(
    input_dataset_name: str,
    game_data_dir: str,
    game_dataset_name: str,
    filter_lang_code: str | None,
) -> tuple[list[int], list[int]]:
    game_data_path = os.path.join(
        game_data_dir, input_dataset_name, game_dataset_name + ".csv"
    )

    source_target_ids = loadtxt(
        game_data_path, dtype=bytes, delimiter=",", skiprows=1
    ).astype(str)
    steps = None
    if source_target_ids.shape[1] >= 3:
        if source_target_ids.shape[1] == 5 and filter_lang_code is not None:
            source_target_ids = array(
                [row for row in source_target_ids if row[4] == filter_lang_code]
            )
            if source_target_ids.shape[0] < 1:
                raise ValueError(
                    "The language code "
                    + filter_lang_code
                    + " doesn't occur in the dataset."
                )
        # The file also contains the number of steps, only load the valid games
        source_target_ids = array([row for row in source_target_ids if int(row[2]) > 0])
        steps = [int(i) for i in source_target_ids[:, 2]]

    source_ids = [int(i) for i in source_target_ids[:, 0]]
    target_ids = [int(i) for i in source_target_ids[:, 1]]
    return source_ids, target_ids, steps


def main():
    args = parse_args()
    data_path = args.data_path
    input_dataset_name = args.input_dataset
    load_data_to_ram = args.load_data_to_ram
    save_n_games = args.save_n_games
    max_games = args.max_games

    game_data_dir, _, output_data_dir = get_data_folders(data_path)
    # Load config file
    config_path = os.path.join(game_data_dir, input_dataset_name, "config.yaml")
    if not os.path.exists(config_path):
        err_msg = f"Config file does not exist: {config_path}"
        raise ValueError(err_msg)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    # Set list of experiments to be run
    if args.experiment_name is not None:
        # Check the given experiment name exists in the config file
        if args.experiment_name not in config["experiments"]:
            err_msg = f"Experiment does not exist: {args.experiment_name}"
            raise ValueError(err_msg)
        experiments = [args.experiment_name]
    else:
        # Running all experiments in config file
        experiments = config["experiments"].keys()

    # If running for one specific model check that it exists in all experiments
    # that are to be run
    if args.model_name is not None:
        for experiment in experiments:
            if args.model_name not in config["experiments"][experiment]["models"]:
                err_msg = f"Model name does not exist in experiment {experiment}: \
                    {args.model_name}"
                raise ValueError(err_msg)
    elif args.model_id is not None:
        for experiment in experiments:
            if not args.model_id < len(config["experiments"][experiment]["models"]):
                err_msg = f"There are fewer models than the model id provided for \
                    experiment {experiment}"
                raise ValueError(err_msg)

    # Create output path if it doesn't already exist
    output_path = os.path.join(output_data_dir, input_dataset_name)
    os.makedirs(output_path, exist_ok=True)

    for experiment in experiments:
        experiment_output_path = os.path.join(output_path, experiment)
        os.makedirs(experiment_output_path, exist_ok=True)
        # Load Agent Tools object - some opitional parameters that might exist in config
        if "from_database" in config["experiments"][experiment]["params"]:
            from_database = config["experiments"][experiment]["params"]["from_database"]
        else:
            from_database = False
        if "use_old_title" in config["experiments"][experiment]["params"]:
            use_old_title = config["experiments"][experiment]["params"]["use_old_title"]
        else:
            use_old_title = False
        tools = load_tools(
            data_path,
            input_dataset_name,
            load_data_to_ram,
            use_old_title,
            from_database,
        )
        for game_dataset in config["experiments"][experiment]["game_datasets"]:
            if "filter_lang" in config["experiments"][experiment]["params"]:
                filter_lang = config["experiments"][experiment]["params"]["filter_lang"]
            else:
                filter_lang = None
            source_ids, target_ids, steps = load_test_dataset(
                input_dataset_name, game_data_dir, game_dataset, filter_lang
            )
            # Create a list of model names to be run
            if args.model_name is not None:
                # One model name was specified in the config file
                models = [args.model_name]
            elif args.model_id is not None:
                models = [config["experiments"][experiment]["models"][args.model_id]]
            else:
                # No model specified in config, so run them all
                models = config["experiments"][experiment]["models"]
            for model in models:
                for start_index in range(0, len(source_ids), max_games):
                    output_filename = (
                        game_dataset + "_" + model + "_" + str(start_index)
                    )
                    output_filename = output_filename.replace("/", "_")
                    output_path = os.path.join(
                        experiment_output_path, output_filename + ".csv"
                    )
                    # We only run this combination if an output file doesn't already
                    # exist
                    if not os.path.exists(output_path):
                        # Save a placeholder file in the output path - will prevent
                        # another batch job running for these games
                        os.makedirs(os.path.join(experiment_output_path), exist_ok=True)
                        savetxt(output_path, [])
                        max_rounds = config["experiments"][experiment]["params"][
                            "max_rounds"
                        ]
                        agent_class = config["experiments"][experiment]["agent_class"]
                        if agent_class == "GreedyEmbeddingAgent":
                            if (
                                "use_titles"
                                in config["experiments"][experiment]["params"]
                            ):
                                use_titles = config["experiments"][experiment][
                                    "params"
                                ]["use_titles"]
                            else:
                                use_titles = False
                            agent = GreedyEmbeddingAgent(
                                tools,
                                max_rounds=max_rounds,
                                model_name=model,
                                use_titles=use_titles,
                            )
                        elif agent_class == "TFIDFAgent":
                            tools_for_corpus = None
                            if (
                                "agent_tools_for_corpus"
                                in config["experiments"][experiment]
                            ):
                                corpus_dataset_name = config["experiments"][experiment][
                                    "agent_tools_for_corpus"
                                ]
                                from_database = False
                                use_old_title = False
                                if (
                                    "params_for_corpus"
                                    in config["experiments"][experiment]
                                ):
                                    if (
                                        "from_database"
                                        in config["experiments"][experiment][
                                            "params_for_corpus"
                                        ]
                                    ):
                                        from_database = config["experiments"][
                                            experiment
                                        ]["params_for_corpus"]["from_database"]
                                    if (
                                        "use_old_title"
                                        in config["experiments"][experiment][
                                            "params_for_corpus"
                                        ]
                                    ):
                                        use_old_title = config["experiments"][
                                            experiment
                                        ]["params_for_corpus"]["use_old_title"]
                                tools_for_corpus = load_tools(
                                    data_path,
                                    corpus_dataset_name,
                                    load_data_to_ram,
                                    use_old_title,
                                    from_database,
                                )
                            agent = TFIDFAgent(
                                tools,
                                max_rounds=max_rounds,
                                lang_iso_code=config["experiments"][experiment][
                                    "params"
                                ]["lang"],
                                manual_seed=config["experiments"][experiment]["params"][
                                    "manual_seed"
                                ],
                                max_articles=config["experiments"][experiment][
                                    "params"
                                ]["max_articles"],
                                agent_tools_for_corpus=tools_for_corpus,
                            )
                        elif agent_class in ("PromptAgent", "ReasoningAgent"):
                            temperature = config["experiments"][experiment]["params"][
                                "temperature"
                            ]
                            top_p = config["experiments"][experiment]["params"]["top_p"]
                            manual_seed = config["experiments"][experiment]["params"][
                                "manual_seed"
                            ]
                            num_retries = config["experiments"][experiment]["params"][
                                "num_retries"
                            ]
                            max_titles = config["experiments"][experiment]["params"][
                                "max_titles"
                            ]
                            if (
                                config["experiments"][experiment]["params"]["llm"]
                                == "gpt"
                            ):
                                gen_model = OpenAIModel(
                                    model, temperature, top_p, manual_seed
                                )
                            elif (
                                config["experiments"][experiment]["params"]["llm"]
                                == "ollama"
                            ):
                                gen_model = OllamaLM(
                                    model,
                                    options={"temperature": temperature},
                                    host_port=args.ollama_host,
                                )
                            if agent_class == "PromptAgent":
                                agent = PromptAgent(
                                    tools,
                                    max_rounds=max_rounds,
                                    model=gen_model,
                                    num_retries=num_retries,
                                    max_titles=max_titles,
                                )
                            elif agent_class == "ReasoningAgent":
                                agent = ReasoningAgent(
                                    tools,
                                    max_rounds=max_rounds,
                                    model=gen_model,
                                    num_retries=num_retries,
                                    max_titles=max_titles,
                                )

                        agent.play_games(
                            source_ids[start_index : start_index + max_games],
                            target_ids[start_index : start_index + max_games],
                            experiment_output_path,
                            output_filename,
                            save_n_games,
                            optimal_steps=steps[start_index : start_index + max_games],
                        )
                        break


if __name__ == "__main__":
    main()
