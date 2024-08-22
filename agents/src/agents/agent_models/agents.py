from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from torch import backends, cuda, device

from agents.agent_models.agent_tools import AgentTools
from agents.utils.models import get_model_from_name
from agents.utils.paths import get_output_dir


class Agent(ABC):
    def __init__(self, agent_tools: AgentTools, max_rounds: int) -> None:
        """
        To do

        Parameters
        ----------
        agent_tools: AgentTools
            to do

        """
        self._tools = agent_tools
        self._games_played: list[dict] = []
        self._max_rounds = max_rounds

    @abstractmethod
    def play_game(
        self, source_page: int | str, target_page: int | str, verbose: bool = False
    ) -> tuple[bool, list[int]]:
        """
        This method will take the source and target page names and
        will implement an approach to find a route between the two
        pages.
        If a route is found then the method returns True and a list of the route found
        (composed of page IDs). If a route is not found then the method returns False
        and a list of the route taken without reaching the target page. A route may not
        be found for serveral reasons including:
        - Needing to take more than the maximum number of rounds the object has
          been set
        - Reaching a page where there are either no links, or no links to which
          the agent has not yet navigated.
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
        """

    def play_games(
        self,
        source_ids: list[int],
        target_ids: list[int],
        output_folder: str,
        output_file_name: str,
    ) -> None:
        """ """
        assert len(source_ids) == len(
            target_ids
        ), "The number of source and target IDs is different"

        for i in range(len(source_ids)):
            self.play_game(source_ids[i], target_ids[i])
            if (i + 1) % 5 == 0:
                self.save_games(output_folder, output_file_name)
        self.save_games(output_folder, output_file_name)

    def _initialise_game(
        self, source_page: int | str, target_page: int | str
    ) -> tuple[int, str, int, str, str]:
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
        tuple[int, str, int, str, str]
            A tuple of the source ID, source title, target ID, target title,
            target summary
        """
        temp_source_id: int | None = (
            self._tools.get_article_id(source_page)
            if type(source_page) is str
            else int(source_page)
        )
        if type(temp_source_id) is int:
            source_id: int = temp_source_id
        else:
            raise ValueError("Source ID not found for " + str(source_page))

        temp_source_title: str | None = (
            self._tools.get_article_title(source_page)
            if type(source_page) is int
            else str(source_page)
        )
        if type(temp_source_title) is str:
            source_title: str = temp_source_title
        else:
            raise ValueError("Source title not found for ID " + str(source_page))

        temp_target_id: int | None = (
            self._tools.get_article_id(target_page)
            if type(target_page) is str
            else int(target_page)
        )
        if type(temp_target_id) is int:
            target_id: int = temp_target_id
        else:
            raise ValueError("Target ID not found for " + str(target_page))

        temp_target_title: str | None = (
            self._tools.get_article_title(target_page)
            if type(target_page) is int
            else str(target_page)
        )
        if type(temp_target_title) is str:
            target_title: str = temp_target_title
        else:
            raise ValueError("Target title not found for ID " + str(target_page))

        target_summary = self._tools.get_article_summary(target_id)

        return (
            int(source_id),
            str(source_title),
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
    ) -> None:
        """ """
        game_summary = {
            "source_id": source_id,
            "source_title": source_title,
            "target_id": target_id,
            "target_title": target_title,
            "found_target": found_target,
            "num_steps": len(route) - 1,
            "route": route,
        }

        self._games_played.append(game_summary)

    def save_games(self, folder_name: str, file_name: str) -> None:
        output_dir = get_output_dir(__file__)
        output_file_path = os.path.join(output_dir, folder_name, file_name + ".csv")
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

        df = pd.DataFrame(self._games_played)

        df.to_csv(output_file_path)


class GreedyEmbeddingAgent(Agent):
    def __init__(
        self, agent_tools: AgentTools, max_rounds: int, model_name: str
    ) -> None:
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
        source_id, source_title, target_id, target_title, target_summary = (
            self._initialise_game(source_page, target_page)
        )
        target_embedding = self._model.encode(target_summary)

        current_id = source_id
        route = [source_id]
        found_target = current_id == target_id
        while (len(route) < self._max_rounds) and not found_target:
            next_page_id = self.play_round(current_id, target_embedding, route)
            if next_page_id == -1:
                break
            found_target = next_page_id == target_id
            route.append(next_page_id)
            current_id = next_page_id

        if verbose:
            self._print_route(source_title, target_title, found_target, route)

        self.record_game(
            source_id, source_title, target_id, target_title, found_target, route
        )

        return found_target, route

    def play_round(
        self, current_page_id: int, target_summary: str | np.ndarray, route: list[int]
    ) -> int:
        forward_summaries = self._tools.get_forward_summaries(current_page_id)
        page_ids = [
            summary_info[0]
            for summary_info in forward_summaries
            if summary_info[0] not in route
        ]
        summaries = [
            summary_info[1]
            for summary_info in forward_summaries
            if summary_info[0] not in route
        ]
        embeddings = self._model.encode(summaries)

        similarities = self._model.similarity(embeddings, target_summary)
        np_similarities = similarities.numpy()
        max_id = np.argmax(np_similarities)
        return page_ids[max_id]


class PromptLLMAgent(Agent):
    def __init__(self, agent_tools: AgentTools, max_rounds: int) -> None:
        super().__init__(agent_tools, max_rounds)
