from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch import backends, cuda, device

from agents.agent_models.generative_lm import GenLM
from agents.agent_models.tools import AgentToolsBase
from agents.utils.models import get_sentence_transformer, get_tfidf_model
from agents.utils.text import pre_process_chinese, pre_process_with_stemming


class Agent(ABC):
    def __init__(self, agent_tools: AgentToolsBase, max_rounds: int) -> None:
        """
        Abstract class representing an Agent capable of playing the wikipedia game.

        Parameters
        ----------
        agent_tools: AgentToolsBase
            An instance of AgentToolsBase that will have been pre-loaded with the data
            with which the game will be played
        max_rounds: int
            The maximum number of rounds that the game can be played for

        """
        self._tools = agent_tools
        self._games_played: list[dict] = []
        self._max_rounds = max_rounds

    @abstractmethod
    def play_game(
        self,
        source_page: int | str,
        target_page: int | str,
        verbose: bool = False,
        optimal_steps: int | None = None,
    ) -> tuple[bool, list[int]]:
        """
        This method will take the source and target page IDs or titles and will
        implement an approach to find a route between the two pages.
        If a route is found then the method returns True and a list of the route found
        (composed of page IDs). If a route is not found then the method returns False
        and a list of the route taken without reaching the target page. A route may not
        be found for serveral reasons including:
        - Needing to take more than the maximum number of rounds the object has
          been set
        - Reaching a page where there are either no links, or no links to which
          the agent has not yet navigated.

        Parameters
        ----------
        source_page: int | str
            Either the source page ID or title
        target_page: int | str
            Either the target page ID or title
        verbose: bool
            If set to True, then the route is printed out
        optimal_steps: int | None
            If not none, then this is the optimal number of steps to solve the game

        Returns
        -------
        tuple[bool, list[int]]
            Returns a boolean that indicates whether a route was found, and a list of
            the route that was taken - the items in the list will be the page IDs
        """

    def play_games(
        self,
        source_ids: list[int],
        target_ids: list[int],
        output_folder: str,
        output_file_name: str,
        save_n_games: int = 100,
        optimal_steps: list[int] | None = None,
    ) -> None:
        """
        This method takes a list of source page IDs, target page IDs, an output
        folder and file name. It plays the game for each pair of source ID and
        target ID and saves the output every save_n_games.

        Parameters
        ----------
        source_ids: list[int]
            A list of page IDs that will act as the start (source) page in each game
        target_ids: list[int]
            A list of page IDs that will act as the end (target) page in each game
        output_folder: str
            The path of the folder to save the results in
        output_file_name: str
            The name of the file to save the results in
        save_n_games: int
            The output will be saved every save_n_games
        optimal_steps: list[int]
            A list of the optimal number of steps for playing each game, can be an
            empty list. If not an empty list, then the optimal number of steps will
            be saved with the output.

        Raises
        ------
        ValueError
            If the number of source and target IDs is different, and if the number
            of steps doesn't match the number of source / target IDs (if the number
            of optimal steps is greater than zero)
        """
        if len(source_ids) != len(target_ids):
            err_msg = "The number of source and target IDs is different."
            raise ValueError(err_msg)

        if optimal_steps is not None and (len(optimal_steps) != len(source_ids)):
            err_msg = "The number of steps doesn't match the number of games"
            raise ValueError(err_msg)

        for i in range(len(source_ids)):
            num_steps = optimal_steps[i] if optimal_steps is not None else None
            self.play_game(source_ids[i], target_ids[i], optimal_steps=num_steps)
            if i % save_n_games == 0:
                # Note that this will save the first record (when i = 0)
                # and then save every save_n_games - saving the first
                # record is deliberate as this is sometimes useful to see
                # that the code has started to run.
                self.save_games(output_folder, output_file_name)
        self.save_games(output_folder, output_file_name)

    def _get_id_title_summary(self, page: int | str) -> tuple[int, str, str]:
        """
        This method accepts either a page ID or a page title and returns the
        corresponding page ID, title and summary (if found)

        Parameters
        ----------
        page: int | str
            Either the ID or the title of a page

        Returns
        -------
        tuple[int, str, str, int, str, str]
            A tuple of the ID, title, and summary

        Raises
        ------
        ValueError
            Raised if the `page` value does not match an ID or title in the data
        """
        temp_id: int | None = (
            self._tools.get_article_id(page) if type(page) is str else int(page)
        )
        if type(temp_id) is int:
            page_id: int = temp_id
        else:
            raise ValueError("ID not found for " + str(page))

        temp_title: str | None = (
            self._tools.get_article_title(page) if type(page) is int else str(page)
        )
        if type(temp_title) is str:
            page_title: str = temp_title
        else:
            raise ValueError("Title not found for ID " + str(page))

        page_summary = str(self._tools.get_article_summary(page_id))

        return page_id, page_title, page_summary

    def _initialise_game(
        self, source_page: int | str, target_page: int | str
    ) -> tuple[int, str, str, int, str, str]:
        """
        When a game is started, either the title or ID for the source and target
        pages can be provided. The titles will probably only be used in a testing
        capacity to examine particular games. This method returns the ID and title
        for both pages, as well as the summary for the target page.

        Parameters
        ----------
        source_page: int | str
            Either the ID or the title of the start page for the game
        target_page: int | str
            Either the ID or the title of the target page for the game

        Returns
        -------
        tuple[int, str, str, int, str, str]
            A tuple of the source ID, source title, source summary, target ID,
            target title, target summary
        """

        source_id, source_title, source_summary = self._get_id_title_summary(
            source_page
        )
        target_id, target_title, target_summary = self._get_id_title_summary(
            target_page
        )

        return (
            int(source_id),
            str(source_title),
            str(source_summary),
            int(target_id),
            str(target_title),
            str(target_summary),
        )

    def _print_route(
        self,
        source_title: str,
        target_title: str,
        found_route: bool,
        route: list[int],
        reasons: list[str] | None = None,
    ) -> None:
        """
        This method takes a route as a list of page IDs and prints the title
        of each page in the route along with a summary of the game and whether
        it was successful

        Parameters
        ----------
        source_title: str
            The title of the start page
        target_title: str
            The title of the target page
        found_route: bool
            A boolean indicating whether a valid route was found
        route: list[int]
            A list of page IDs
        reasons: list[str]
            A list of reasons - optional
        """
        print("Game to travel from `" + source_title + "` to `" + target_title + "`")
        if found_route:
            print("Successful")
        else:
            if len(route) == self._max_rounds:
                print("Failed - max rounds reached")
            else:
                print("Failed - other reason")
        print("Route taken:")
        for id, page in enumerate(route):
            page_title = self._tools.get_article_title(page) if page > -1 else "FAILURE"
            if reasons is not None:
                print(id, ":", page_title, "-", reasons[id])
            else:
                print(id, ":", page_title)

    def record_game(
        self,
        source_id: int,
        source_title: str,
        target_id: int,
        target_title: str,
        found_target: bool,
        route: list,
        similarity: float | None = None,
        optimal_steps: int | None = None,
        err_msg: str = "",
        additional_fields: dict | None = None,
    ) -> None:
        """
        This method adds the game details provided to the list of played games

        Parameters
        ----------
        source_id: int
            The ID of the start (source) page
        source_title: str
            The title of the source page
        target_id: int
            The ID of the end (target) page
        target_title: str
            The title of the target page
        found_target: bool
            If the game was successful in finding the target page
        route: list
            A list of page IDs representing the route taken
        similarity: float | None
            A value representing the similiarity of the source and target pages - this
            will not be provided for all types of agents
        optimal_steps: int | None
            If not none, then this is the optimal number of steps to solve the game
        additional_fields: dict
            Stores the value of any non-standard fields to be recorded
        """
        num_steps = len(route) - 2 if route[-1] == -1 else len(route) - 1
        game_summary = {
            "source_id": source_id,
            "source_title": source_title,
            "target_id": target_id,
            "target_title": target_title,
            "found_target": found_target,
            "num_steps": num_steps,
            "error_message": err_msg,
        }
        if similarity is not None:
            game_summary["similarity"] = similarity
        if optimal_steps is not None:
            game_summary["optimal_steps"] = optimal_steps
        if additional_fields is not None:
            for field in additional_fields:
                game_summary[field] = additional_fields[field]
        game_summary["route"] = route

        self._games_played.append(game_summary)

    def save_games(self, folder_path: str, file_name: str) -> None:
        """
        This method saves all the games recorded in the instance.

        Parameters
        ----------
        folder_name: str
            The name of the folder to save the results
        file_name: str
            The name of the file to save the results
        """
        output_file_path = os.path.join(folder_path, file_name + ".csv")
        os.makedirs(os.path.join(folder_path), exist_ok=True)

        df = pd.DataFrame(self._games_played)

        df.to_csv(output_file_path, sep="\t")

    def get_games_played(self) -> list[dict]:
        """
        This method returns the list of dictionaries containing the games
        played by the agent. Expected to use this for testing only.

        Returns
        -------
        list[dict]
            List of dictionaries, each dictionary contains details of a game:
            "source_id" - the id of the source page
            "source_title" - the title of the source page
            "target_id" - the id of the target page
            "target_title" - the title of the target page
            "similarity" - the similarity of the source and target (may be 0 if the
                    agent doesn't calculate a similarity)
            "found_target" - True if reached target page, False otherwise
            "num_steps" - the number of steps taken in the game (length of route - 1)
            "route" - a list of page ids representing the route taken in the game, will
                    include the source and target page ids
        """
        return self._games_played


class TFIDFAgent(Agent):
    def __init__(
        self,
        agent_tools: AgentToolsBase,
        max_rounds: int,
        lang_iso_code: str,
        manual_seed: int = 42,
        max_articles: int = 100000,
        agent_tools_for_corpus: AgentToolsBase | None = None,
    ) -> None:
        """
        Class inherits from Agent and plays the wikipedia game by using a TF-IDF
        model. The corpus used for the model can be created using the same dataset as
        used to play the game, and there is also be an option for a different dataset
        to be used as the corpus.

        Parameters
        ----------
        agent_tools: AgentToolsBase
            An instance of AgentToolsBase that will have been pre-loaded with the data
            with which the game will be played
        max_rounds: int
            The maximum number of rounds that the game can be played for
        lang_iso_code: str
            The iso code for the language to be used to create the corpus
        manual_seed: int
            The seed to use to select articles to use in the corpus
        max_articles: int
            The maximum number of articles to use to create the corpus
        agent_tools_for_corpus: AgentToolsBase | None
            If None, then the agent_tools object is used to create the corpus
        """
        super().__init__(agent_tools, max_rounds)

        self._lang = lang_iso_code
        if agent_tools_for_corpus is None:
            # If no object is passed as an argument, then set the agent's own agent
            # tools object to be used to create the corpus as well.
            agent_tools_for_corpus = agent_tools
        # Set the function to be used to process the text, both for the TF-IDF model
        # and also when creating the vectors from the article summaries. Chinese is
        # currently different to other languages we process as it doesn't require
        # stemming.
        if lang_iso_code == "zh":
            self._preprocess_func = pre_process_chinese
        else:
            self._preprocess_func = pre_process_with_stemming
        # Sets the model, vocab and dictionary objects - all of these are needed when
        # creating a vector of each summary article.
        self._model, self._dictionary = get_tfidf_model(
            agent_tools_for_corpus,
            manual_seed,
            max_articles,
            self._preprocess_func,
            self._lang,
        )

    def _vectorise(self, text: str) -> tuple[np.array, int, int]:
        """
        This method takes a string of text (expected to be some summary text about
        an article or page) and returns an array the same length as the vocab vector.

        Parameters
        ----------
        text: str
            The text to be processed

        Returns
        -------
        np.array
            A numpy array representing the text in vector form.
        int
            The number of words after the text has been processed
        int
            The number of words that did not match a value in the vocabulary
        """
        list_words = self._preprocess_func(text, self._lang)
        bow = self._dictionary.doc2bow(list_words)
        tfidf_output = self._model[bow]
        vector = np.zeros(len(self._dictionary.values()))
        for id, freq in tfidf_output:
            vector[id] = freq
        return vector, len(list_words), len(list_words) - len(bow)

    def _similarity(
        self, candidate_vectors: list[np.array], target_vector: np.array
    ) -> np.array:
        """
        This method calculates the similarity between a list of candidate
        vectors and the vector from the target page using the cosine similarity.

        Parameters
        ----------
        candidate_vctors: list[np.array]
            List of arrays representing the links on the current page
        target_vector: np.array
            Array representing the target page

        Returns
        -------
        np.array
            The cosine similarity value between the candidate vectors and target vector
        """
        return cosine_similarity(candidate_vectors, [target_vector])

    def play_game(
        self,
        source_page: int | str,
        target_page: int | str,
        verbose: bool = False,
        optimal_steps: int | None = None,
    ) -> tuple[bool, list[int]]:
        """
        This method takes the source and target page IDs or titles and implements a
        greedy approach to find a route between the two pages. The approach considers
        the TF-IDF vector of each of the summaries on the current page, and chooses the
        one that is closest to the target page.

        If a route is found then the method returns True and a list of the route found
        (composed of page IDs). If a route is not found then the method returns False
        and a list of the route taken without reaching the target page. A route may not
        be found for serveral reasons including:
        - Needing to take more than the maximum number of rounds the object has
          been set
        - Reaching a page where there are either no links, or no links to which
          the agent has not yet navigated.

        Parameters
        ----------
        source_page: int | str
            Either the source page ID or title
        target_page: int | str
            Either the target page ID or title
        verbose: bool
            If set to True, then the route is printed out
        optimal_steps: int | None
            If not none, then this is the optimal number of steps to solve the game

        Returns
        -------
        tuple[bool, list[int]]
            Returns a boolean that indicates whether a route was found, and a list of
            the route that was taken - the items in the list will be the page IDs
        """
        (
            source_id,
            source_title,
            source_summary,
            target_id,
            target_title,
            target_summary,
        ) = self._initialise_game(source_page, target_page)

        source_embedding = self._vectorise(source_summary)[0]
        target_embedding = self._vectorise(target_summary)[0]
        similarity = self._similarity([source_embedding], target_embedding)[0]

        current_id = source_id
        route = [source_id]
        found_target = current_id == target_id
        # Keep a count of total words and oov words encountered during the game.
        total_words = 0
        total_oov_words = 0
        while (len(route) <= self._max_rounds) and not found_target:
            next_page_id, num_words, num_oov = self._play_round(
                current_id, route, target_embedding
            )
            if next_page_id == -1:
                break
            found_target = next_page_id == target_id
            route.append(next_page_id)
            current_id = next_page_id
            total_words += num_words
            total_oov_words += num_oov

        if verbose:
            self._print_route(source_title, target_title, found_target, route)

        self.record_game(
            source_id,
            source_title,
            target_id,
            target_title,
            found_target,
            route,
            similarity,
            optimal_steps,
            additional_fields={
                "total_words": total_words,
                "total_oov_words": total_oov_words,
            },
        )

        return found_target, route

    def _play_round(
        self,
        current_page_id: int,
        route: list[int],
        target_summary: np.ndarray,
    ) -> tuple[int, int, int]:
        """
        Given a current page id, this method gets all the links that exist on the
        page. It compares the links to the target summary and returns the ID of the
        page to navigate to next. This implementation selects the next page by
        calculating the similarity of the target summary TF-IDF vector to the summary
        of each possible link page - the link that has the most similar summary is
        the one that is selected.

        Parameters
        ----------
        current_page_id: int
            The ID of the current page
        route: list[int]
            The rotue taken so far in the current game.
        target_summary: array
            An embedding of the summary of the target page.

        Returns
        -------
        int
            The ID of the page to navigate to next
        int
            The number of words encountered
        int
            The number of out-of-vocabularly words
        """
        forward_text = self._tools.get_forward_summaries(current_page_id)
        if len(forward_text) > 0:
            page_ids = [
                page_info[0]
                for page_info in forward_text
                if (page_info[0] not in route) and (page_info[1] is not None)
            ]
            page_text = [
                page_info[1]
                for page_info in forward_text
                if (page_info[0] not in route) and (page_info[1] is not None)
            ]
            if len(page_text) > 0:
                embeddings = []
                num_words = []
                num_oov = []
                for text in page_text:
                    embedding, words, oov = self._vectorise(str(text))
                    embeddings.append(embedding)
                    num_words.append(words)
                    num_oov.append(oov)

                similarities = self._similarity(embeddings, target_summary)
                max_id = np.argmax(similarities)
                return page_ids[max_id], sum(num_words), sum(num_oov)
            return -1, 0, 0
        return -1, 0, 0


class GreedyEmbeddingAgent(Agent):
    def __init__(
        self,
        agent_tools: AgentToolsBase,
        max_rounds: int,
        model_name: str,
        use_titles: bool = False,
    ) -> None:
        """
        Class inherits from Agent and plays the wikipedia game by comparing the
        similarity of summary embeddings of links on the current page and the
        target page. The algorithm selects the link that has a summary or title that
        is scored as being most similar to the target page summary.

        Parameters
        ----------
        agent_tools: AgentToolsBase
            An instance of AgentToolsBase that will have been pre-loaded with the data
            with which the game will be played
        max_rounds: int
            The maximum number of rounds that the game can be played for
        model_name: str
            The name of the sentence transformer model to be used for embeddings
        use_titles: bool
            If set to True, the page / link titles will be used instead of summaries.
            The summaries will be used by default.
        """
        super().__init__(agent_tools, max_rounds)

        # Set device
        if cuda.is_available():
            self.device = device("cuda:0")
        elif backends.mps.is_available():
            self.device = device("mps")
        else:
            self.device = "cpu"

        self._model = get_sentence_transformer(model_name).to(self.device)
        self._use_titles = use_titles

    def play_game(
        self,
        source_page: int | str,
        target_page: int | str,
        verbose: bool = False,
        optimal_steps: int | None = None,
    ) -> tuple[bool, list[int]]:
        """
        This method takes the source and target page IDs or titles and implements a
        greedy approach to find a route between the two pages. The approach considers
        the embedding of each of the summaries on the current page, and choses the one
        that is closest to the target page. The cosine distance is set by default
        when creating the sentence transformer model.

        If a route is found then the method returns True and a list of the route found
        (composed of page IDs). If a route is not found then the method returns False
        and a list of the route taken without reaching the target page. A route may not
        be found for serveral reasons including:
        - Needing to take more than the maximum number of rounds the object has
          been set
        - Reaching a page where there are either no links, or no links to which
          the agent has not yet navigated.

        Parameters
        ----------
        source_page: int | str
            Either the source page ID or title
        target_page: int | str
            Either the target page ID or title
        verbose: bool
            If set to True, then the route is printed out
        optimal_steps: int | None
            If not none, then this is the optimal number of steps to solve the game

        Returns
        -------
        tuple[bool, list[int]]
            Returns a boolean that indicates whether a route was found, and a list of
            the route that was taken - the items in the list will be the page IDs
        """
        (
            source_id,
            source_title,
            source_summary,
            target_id,
            target_title,
            target_summary,
        ) = self._initialise_game(source_page, target_page)
        source_embedding = self._model.encode(source_summary)
        target_embedding = self._model.encode(target_summary)
        similarity = float(self._model.similarity(source_embedding, target_embedding))

        current_id = source_id
        route = [source_id]
        found_target = current_id == target_id
        while (len(route) <= self._max_rounds) and not found_target:
            next_page_id = self._play_round(
                current_id, route, target_embedding, self._use_titles
            )
            if next_page_id == -1:
                break
            found_target = next_page_id == target_id
            route.append(next_page_id)
            current_id = next_page_id

        if verbose:
            self._print_route(source_title, target_title, found_target, route)

        self.record_game(
            source_id,
            source_title,
            target_id,
            target_title,
            found_target,
            route,
            similarity,
            optimal_steps,
        )

        return found_target, route

    def _play_round(
        self,
        current_page_id: int,
        route: list[int],
        target_summary: np.ndarray,
        use_titles: bool,
    ) -> int:
        """
        Given a current page id, this method gets all the links that exist on the
        page. It compares the links to the target summary and returns the ID of the
        page to navigate to next. This implementation selects the next page by
        calculating the similarity of the target summary embedding to the summary
        of each possible link page - the link that has the most similar summary is
        the one that is selected.

        Parameters
        ----------
        current_page_id: int
            The ID of the current page
        route: list[int]
            The rotue taken so far in the current game.
        target_summary: array
            An embedding of the summary of the target page.
        use_titles: bool
            If set to True, then page / link titles will be used instead of the page
            summaries

        Returns
        -------
        int
            The ID of the page to navigate to next
        """
        if use_titles:
            forward_text = self._tools.get_forward_titles(current_page_id)
        else:
            forward_text = self._tools.get_forward_summaries(current_page_id)
        if len(forward_text) > 0:
            page_ids = [
                page_info[0]
                for page_info in forward_text
                if (page_info[0] not in route) and (page_info[1] is not None)
            ]
            page_text = [
                page_info[1]
                for page_info in forward_text
                if (page_info[0] not in route) and (page_info[1] is not None)
            ]
            if len(page_text) > 0:
                embeddings = self._model.encode(page_text)

                similarities = self._model.similarity(embeddings, target_summary)
                np_similarities = similarities.numpy()
                max_id = np.argmax(np_similarities)
                return page_ids[max_id]
            return -1
        return -1


class PromptAgent(Agent):
    def __init__(
        self,
        agent_tools: AgentToolsBase,
        max_rounds: int,
        model: GenLM,
        num_retries: int = 1,
        max_titles: int = -1,
    ) -> None:
        """
        Class inherits from Agent and plays the wikipedia game by giving a
        prompt to an OpenAI GPT model with the

        Parameters
        ----------
        agent_tools: AgentToolsBase
            An instance of AgentToolsBase that will have been pre-loaded with the data
            with which the game will be played
        max_rounds: int
            The maximum number of rounds that the game can be played for
        model: GenLM
            An object derived from GenLM which has the functionality of a generative
            LLM to select a title from a given list.
        num_retries: (int, optional)
            The number of times the agent will retry to select a title if the title
            selected is not in the list of titles provided. Defaults to 1.
        max_titles: (int, optional)
            The maximum number of titles to provide to the model. Defaults to -1.

        Raises
        ------
        ValueError
            If the temperature is not a valid value, must be between 0 and 1.
        """
        super().__init__(agent_tools, max_rounds)

        self._model = model
        self._num_retries = num_retries
        self._max_titles = max_titles

        self._system_content = (
            "You are a helpful assistant that helps me to play the Wikipedia Game! "
        )
        self._query1_content = (
            "If I give you the title and a summary of the target page that I am "
            "trying to navigate to, can you help me navigate to the target page? "
        )
        self._response1_content = "Of course! What is your target page?"
        self._response2_content = (
            "Sure! I can help you navigate to that page, what titles are there on "
            "your current page?"
        )
        self._play_round_content = (
            "Here is a list of titles, which one should I pick? Please provide your "
            'answer in JSON format with the selected title given with key "title" and '
            'a justification given with key "reason". The selected title must be from '
            "the following list: "
        )

    def play_game(
        self,
        source_page: int | str,
        target_page: int | str,
        verbose: bool = False,
        optimal_steps: int | None = None,
    ) -> tuple[bool, list[int]]:
        """
        This method takes the source and target page IDs or titles and implements a
        prompting approach to find a route between the two pages. The approach presents
        a LLM model with a summary of the target page and a list of the titles on the
        current page and asks the model to pick a title.

        If a route is found then the method returns True and a list of the route found
        (composed of page IDs). If a route is not found then the method returns False
        and a list of the route taken without reaching the target page. A route may not
        be found for serveral reasons including:
        - Needing to take more than the maximum number of rounds the object has
          been set
        - Reaching a page where there are either no titles, or no titles to which
          the agent has not yet navigated.

        Parameters
        ----------
        source_page: int | str
            Either the source page ID or title
        target_page: int | str
            Either the target page ID or title
        verbose: bool
            If set to True, then the route is printed out
        optimal_steps: int | None
            If not none, then this is the optimal number of steps to solve the game

        Returns
        -------
        tuple[bool, list[int]]
            Returns a boolean that indicates whether a route was found, and a list of
            the route that was taken - the items in the list will be the page IDs
        """
        (
            source_id,
            source_title,
            source_summary,
            target_id,
            target_title,
            target_summary,
        ) = self._initialise_game(source_page, target_page)

        game_messages = self._set_messages(target_title, target_summary)

        current_id = source_id
        route = [source_id]
        reasons = ["Start page"]
        found_target = current_id == target_id
        while (len(route) <= self._max_rounds) and not found_target:
            next_page_id, reason = self._play_round(
                current_id, route, game_messages, target_title
            )
            route.append(next_page_id)
            reasons.append(reason)
            if next_page_id == -1:
                break
            found_target = next_page_id == target_id
            current_id = next_page_id

        if len(route) > self._max_rounds and not found_target:
            reasons[-1] = "Maximum number of steps reached"
        if verbose:
            self._print_route(source_title, target_title, found_target, route, reasons)

        self.record_game(
            source_id,
            source_title,
            target_id,
            target_title,
            found_target,
            route,
            optimal_steps=optimal_steps,
            err_msg=reasons[-1] if not found_target else "",
        )

        return found_target, route

    def _set_messages(self, target_title: str, target_summary: str) -> list[dict]:
        """
        Set the initial messages to send at the start of each round

        Parameters
        ----------
        target_title: str
            The title of the target page
        target_summary: str
            The summary of the target page

        Returns
        -------
        list[dict]
            A list of dictionaries containing the role and content to be used
            at the start of the prompt for the given game.
        """

        message1 = {"role": "system", "content": self._system_content}

        message2 = {"role": "user", "content": self._query1_content}

        message3 = {"role": "assistant", "content": self._response1_content}

        content = '{"title": "' + target_title + '", '
        content += '"summary": "' + target_summary + '"}'

        message4 = {"role": "user", "content": content}

        message5 = {"role": "assistant", "content": self._response2_content}

        return [message1, message2, message3, message4, message5]

    def _play_round(
        self,
        current_page_id: int,
        route: list[int],
        game_messages: list[dict],
        target_title: str,
    ) -> tuple[int, str]:
        """
        Given a current page id, this method gets all the titles of the titles that
        exist on the page. It presentes the title titles to the LLM along with a summary
        of the target page and asks the LLM to select a title.
        It then returns the ID of the selected title

        Parameters
        ----------
        current_page_id: int
            The ID of the current page
        route: list[int]
            The rotue taken so far in the current game.
        messages: list[dict]
            A list of dictionaries containing the messages to be used at the start of
            the prompt - these are the same for all rounds in the game.
        target_title: str
            The title of the target page to provide in error messages.

        Returns
        -------
        int
            The ID of the page to navigate to next
        """
        forward_titles = self._tools.get_forward_titles(current_page_id)
        titles: list[str] = [
            str(forward_title[1])
            for forward_title in forward_titles
            if (forward_title[0] is not None)
            and (forward_title[1] is not None)
            and (forward_title[0] not in route)
        ]
        ids: list[int] = [
            int(forward_title[0])
            for forward_title in forward_titles
            if (forward_title[0] is not None)
            and (forward_title[1] is not None)
            and (forward_title[0] not in route)
        ]
        if len(titles) != len(ids):
            # If this is True, then something has gone very wrong! Throw an error.
            err_msg = (
                "Error creating list of IDs and Titles, they are different lengths!"
            )
            raise ValueError(err_msg)
        if len(titles) > 0:
            content = " "
            for title in titles[: self._max_titles]:
                content += '"' + title + '", '
            content = "[" + content[:-2] + "]"

            message = {"role": "user", "content": self._play_round_content + content}

            messages = game_messages.copy()
            messages.append(message)
            for m in messages:
                if m["content"] is None:
                    return -1, ""

            response = self._model.get_response(messages)

            retry = 0
            err_msg = ""
            while retry <= self._num_retries:  # allow for retries
                if response is None:
                    # Will be 'None' if the content filter interferes, simulate a
                    # response and retry
                    retry += 1
                    response = (
                        '{\n  "title": "Unknown title",\n "reason": "'
                        'I did not know which title to select."\n}'
                    )
                    content_retry = (
                        'But "Unknown title" is not in the list '
                        "that was provided! You must select a title from the "
                        "list of titles. Please try again and provide your "
                        "answer in JSON format with the selected title given "
                        'with key "title" and a justification given with key "reason". '
                        "The JSON should only include the two fields 'title' and "
                        "'reason', and the title should only include the title text. "
                        "You are tying to select the best page title to reach "
                        f"the target page '{target_title}'."
                    )
                elif response == "BadRequestError":
                    return -1, response
                else:
                    # There is a response
                    try:
                        json_response = json.loads(response)
                    except (json.decoder.JSONDecodeError, RecursionError):
                        json_response = None
                    if json_response is None:
                        # Some error when loading the response to JSON - making the
                        # assumption that the recursion error occurs when decoding
                        # the JSON object (which is what is happening so far)
                        retry += 1
                        err_msg = "Response not in valid JSON format"
                        content_retry = (
                            "You did not provide your response in JSON "
                            "format - please try again!"
                            "The JSON should only include the two fields 'title' and "
                            "'reason', and the title should only include the title "
                            "text. "
                            "You are tying to select the best page title to reach "
                            f"the target page '{target_title}'."
                        )
                    else:
                        # We have a valid json object
                        if "title" in json_response:
                            next_link = json_response["title"]
                            if next_link in titles:
                                # Excellent - no hallucinations!
                                list_id = titles.index(next_link)
                                # list_id should always be a valid index of the ids
                                # list as there was a check above to make sure that
                                # the titles and ids lists were the same length and
                                # no items are removed from the list during this process
                                next_link_id = ids[list_id]
                                if next_link_id is None:
                                    # This error shouldn't be reached?
                                    err_msg = (
                                        "Couldn't find article ID for " + next_link
                                    )
                                    return -1, err_msg
                                if "reason" in json_response:
                                    return next_link_id, json_response["reason"]
                                return next_link_id, "No reason provided"
                            # *sigh* - hallucination
                            retry += 1
                            err_msg = "Hallucination - title not in list"
                            content_retry = (
                                f'But "{next_link}" is not in the list '
                                "that was provided! You must select a title from the "
                                "list of titles. Please try again and provide your "
                                "answer in JSON format with the selected title given "
                                'with key "title" and a justification given with key '
                                '"reason".'
                                "To jog your memory, 10 example titles from the "
                                f"list are: {np.random.choice(titles, 10)}. "
                                "The JSON should only include the two fields 'title' "
                                "and 'reason', and the title should only include the "
                                "title text. "
                                "You are tying to select the best page title to reach "
                                f"the target page '{target_title}'."
                            )
                        else:
                            # No link in the response
                            retry += 1
                            err_msg = "No title provided in response"
                            content_retry = (
                                "You must provide a title to navigate to. Please try "
                                "again and provide your answer in JSON format with the "
                                'selected title given with key "title" and a '
                                'justification given with key "reason". '
                                "The JSON should only include the two fields 'title' "
                                "and 'reason', and the title should only include the "
                                "title text. "
                                "You are tying to select the best page title to reach "
                                f"the target page '{target_title}'."
                            )

                message_response = {"role": "assistant", "content": response}
                messages.append(message_response)
                message_retry = {"role": "user", "content": content_retry}
                messages.append(message_retry)

                response = self._model.get_response(messages)
            # The agent was not able to select a link
            return -1, err_msg
        # There are no links from the current page
        return -1, "Ran out of titles to pick"
