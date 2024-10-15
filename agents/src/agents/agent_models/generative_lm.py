from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from openai import AzureOpenAI, BadRequestError


class GenLM(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_response(self, messages: list[dict[Any, Any]]) -> str | None:
        """
        To do
        """


class MockGenLM(GenLM):
    def __init__(self, test_responses: list[dict]) -> None:
        self._test_responses = test_responses

    def get_response(self, messages: list[dict[Any, Any]]) -> str | None:
        if len(self._test_responses) > 0:
            response = self._test_responses.pop(0)
            if response["type"] == "json":
                link = response["link"]
                return (
                    '{\n "link": "'
                    + link
                    + '",\n "reason": "'
                    + messages[-1]["content"][0]
                    + '"\n}'
                )
        return "not valid json"


class OpenAIModel(GenLM):
    def __init__(
        self,
        deployment_name: str,
        temperature: float = 0.2,
        top_p: float = 0.1,
        seed: int = 42,
    ) -> None:
        self._deployment_name = deployment_name
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed

        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def get_response(self, messages: list[dict[Any, Any]]) -> str | None:
        try:
            completion = self._client.chat.completions.create(
                model=self._deployment_name,
                messages=messages,
                temperature=self._temperature,
                top_p=self._top_p,
                seed=self._seed,
                response_format={"type": "json_object"},
            )

            return completion.choices[0].message.content
        except BadRequestError:
            return "BadRequestError"


class HumanLM(GenLM):
    def __init__(self) -> None:
        """This is class that allows a human to respond to the messages."""

    def get_response(self, messages: list[dict[Any, Any]]) -> str | None:
        for message in messages:
            print(f"{message['role']}: {message['content']}")
        return input("Human: ")
