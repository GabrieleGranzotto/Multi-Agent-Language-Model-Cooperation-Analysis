import requests, re, ollama
from typing import Optional

class OllamaAgent:
    def __init__(self, model_name: str = "gemma4"):
        self.model_name = model_name
        self.message = []

    def _add_message(self, role: str, content: str):
        # append a message to the conversation history
        # valid roles are "system", "user", and "assistant"
        self.message.append({"role": role, "content": content})

    def respond(self):
        response = ollama.chat(model = self.model_name, messages = self.message)
        self._add_message("assistant", response["message"]["content"])

    def listen(self, user: str = "AGENT", prompt: str = None):
        """
        Listen for a user's input and add it to the conversation history.

        Args:
            user (str): The name of the user providing input. Options are "AGENT" or "JUDGE". Defaults to "AGENT".
            prompt (str): The input provided by the user. If None, the function will wait for user input.
        """
        self._add_message("user", f"[{user}]: {prompt}")

    def _context_definition(self, game: str, actions: list):
        """
        Define the context for the agent's behavior and capabilities.
        """
        context = f"""
You are an AI agent designed to play a classic coordination games (in this specific context: {game}) and interact with another AI agent.
You start chatting with the other agent called [AGENT] about the intention of the game and agree on a strategy to play the game.
Then you will not be able to communicate with the other agent anymore and you will have to play the game.
Another user called [JUDGE] will you asking about your strategy and the game you are playing. 
You will have to answer the questions of the [JUDGE] in a standardized way.
The answer should be in the following format:
{{"Answer": [ANSWER], "Explanation": <explanation of the answer>}}
The options for the [ANSWER] are mandatory: [{actions[0]}] or [{actions[1]}].
        """
        self._add_message("system", context)