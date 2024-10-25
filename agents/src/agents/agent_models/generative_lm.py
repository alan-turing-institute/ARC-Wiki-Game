from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import ollama
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
                link = response["title"]
                return (
                    '{\n "title": "'
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


class OllamaLM(GenLM):
    def __init__(
        self,
        model: str,
        options: dict | None = None,
        keep_alive: int = 300,
        host_port: int = 11434,
    ):
        """This model calls the Ollama API to get responses.

        To use call the get_response method with a list of messages.

        Args:
            model (str): The name of the model to use.
            options (dict | None, optional): The parameters to be passed to the ollama
                model. Defaults to None. The default includes setting the temperature
                to 0.0.
            keep_alive (int, optional): The number of seconds to keep the model loaded
                to RAM. Defaults to 300.
        """
        if options is None:
            options = {"temperature": 0.0}
        self.options = options

        self.keep_alive = keep_alive

        if model not in [mod["name"] for mod in ollama.list()["models"]]:
            print("Model not found. Trying to download...")
            ollama.pull(model)

        self.model = model

        self.ollama_client = ollama.Client(host=f"localhost:{host_port:d}")

    def get_response(self, messages: list[dict[Any, Any]]) -> str | None:
        """Get the response from the Ollama model.

        This will be in a JSON format.

        Args:
            messages (list[dict[Any, Any]]): The list of messages to send to the model.
                They are un the format {"role": "user", "content": "message"}.

        Returns:
            str | None: This is the response from the model. It will only include the
                content of the message. Should be in JSON format.
        """
        return self.ollama_client.chat(
            self.model,
            messages=messages,
            format="json",
            options=self.options,
            keep_alive=self.keep_alive,
        )["message"]["content"]
