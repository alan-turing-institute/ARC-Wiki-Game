from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from torch import backends, cuda, device

from agents.agent_models.tools import AgentToolsBase
from agents.utils.models import get_model_from_name


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
        self, source_page: int | str, target_page: int | str, verbose: bool = False
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

        Returns
        -------
        tuple[bool, list[int]]
            Returns a boolean that indicates whether a route was found, and a list of
            the route that was taken - the items in the list will be the page IDs
        """

    @abstractmethod
    def play_round(
        self,
        current_page_id: int,
        target_summary: str | np.ndarray,
        route: list[int],
    ) -> int:
        """
        Given a current page id, this method gets all the links that exist on the
        page. It compares the links to the target summary and returns the ID of the
        page to navigate to next. The exact implementation will depend on the child
        class.

        Parameters
        ----------
        current_page_id: int
            The ID of the current page
        target_summary: str | array
            Either a string summary or embedding of the summary of the target page.
        route: list[int]
            The rotue taken so far in the current game.

        Returns
        -------
        int
            The ID of the page to navigate to next
        """

    def play_games(
        self,
        source_ids: list[int],
        target_ids: list[int],
        output_folder: str,
        output_file_name: str,
        save_n_games: int = 100,
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
            The folder to save the results in
        output_file_name: str
            The name of the file to save the results in
        save_n_games: int
            The output will be saved every save_n_games
        """
        assert len(source_ids) == len(
            target_ids
        ), "The number of source and target IDs is different"

        for i in range(len(source_ids)):
            self.play_game(source_ids[i], target_ids[i])
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
        self, source_title: str, target_title: str, found_route: bool, route: list[int]
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
            page_title = self._tools.get_article_title(page)
            print(id, ":", page_title)

    def record_game(
        self,
        source_id: int,
        source_title: str,
        target_id: int,
        target_title: str,
        found_target: bool,
        route: list,
        similarity: float = 0,
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
        similarity: float
            A value representing the similiarity of the source and target pages - this
            will not be provided for all types of agents
        """
        game_summary = {
            "source_id": source_id,
            "source_title": source_title,
            "target_id": target_id,
            "target_title": target_title,
            "similarity": similarity,
            "found_target": found_target,
            "num_steps": len(route) - 1,
            "route": route,
        }

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


class GreedyEmbeddingAgent(Agent):
    def __init__(
        self, agent_tools: AgentToolsBase, max_rounds: int, model_name: str
    ) -> None:
        """
        Class inherits from Agent and plays the wikipedia game by comparing the
        similarity of summary embeddings of links on the current page and the
        target page. The algorithm selects the link that has a summary that
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
        """
        super().__init__(agent_tools, max_rounds)

        # Set device
        if cuda.is_available():
            self.device = device("cuda:0")
        elif backends.mps.is_available():
            self.device = device("mps")
        else:
            self.device = "cpu"

        self._model = get_model_from_name(model_name).to(self.device)

    def play_game(
        self, source_page: int | str, target_page: int | str, verbose: bool = False
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
            next_page_id = self.play_round(current_id, target_embedding, route)
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
        )

        return found_target, route

    def play_round(
        self, current_page_id: int, target_summary: str | np.ndarray, route: list[int]
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
        target_summary: str | array
            Either a string summary or embedding of the summary of the target page.
        route: list[int]
            The rotue taken so far in the current game.

        Returns
        -------
        int
            The ID of the page to navigate to next
        """
        forward_summaries = self._tools.get_forward_summaries(current_page_id)
        if len(forward_summaries) > 0:
            page_ids = [
                summary_info[0]
                for summary_info in forward_summaries
                if (summary_info[0] not in route) and (summary_info[1] is not None)
            ]
            summaries = [
                summary_info[1]
                for summary_info in forward_summaries
                if (summary_info[0] not in route) and (summary_info[1] is not None)
            ]
            if len(summaries) > 0:
                embeddings = self._model.encode(summaries)

                similarities = self._model.similarity(embeddings, target_summary)
                np_similarities = similarities.numpy()
                max_id = np.argmax(np_similarities)
                return page_ids[max_id]
            return -1
        return -1
