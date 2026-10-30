import requests, re, ollama
from typing import Optional

class OllamaAgent:
    def __init__(self, game, model_name: str = "gemma4"):
        self.model_name = model_name
        self.game = game
        self.messages = []
        self._context_definition()

    def _add_message(self, role: str, content: str):
        # append a message to the conversation history
        # valid roles are "system", "user", and "assistant"
        self.messages.append({"role": role, "content": content})

    def respond(self):
        response = ollama.chat(model = self.model_name, messages = self.messages)
        content = response["message"]["content"]
        self._add_message("assistant", content)
        return content

    def listen(self, user: str = "AGENT", prompt: str = None):
        """
        Listen for a user's input and add it to the conversation history.

        Args:
            user (str): The name of the user providing input. Options are "AGENT" or "JUDGE". Defaults to "AGENT".
            prompt (str): The input provided by the user. If None, the function will wait for user input.
        """
        if prompt == None:
            raise ValueError("Prompt cannot be None. Please provide a valid input.")
        self._add_message("user", f"[{user}]: {prompt}")

    def _context_definition(self):
        """
        Define the context for the agent's behavior and capabilities.
        """
        context = f"""
You are an AI agent designed to play a classic coordination games (in this specific context: {self.game.name}) and interact with another AI agent.
You start chatting with the other agent called [AGENT] about the intention of the game and agree on a strategy to play the game.
Then you will not be able to communicate with the other agent anymore and you will have to play the game.
Another user called [JUDGE] will you asking about your strategy and the game you are playing. 
You will have to answer the questions of the [JUDGE] in a standardized way.
The answer should be in the following format:
{{"Answer": [ANSWER], "Explanation": <explanation of the answer>}}
The options for the [ANSWER] are mandatory: [{self.game.actions[0]}] or [{self.game.actions[1]}].
The <Explanation> should be a concise explanation of the reasoning behind the chosen answer.
Desctiption of the game: {self.game.description}
        """
        self._add_message("system", context)

    def get_strategy(self):
        """
        Extract the strategy from the agent's last response.
        """
        user = "JUDGE"
        final_message = f"""
[{user}]: Now the other agent can't listen to you. Please provide your strategy.
Take in consideration the previous conversation you had with the other agent, included the coordination strategy you agreed on.
Take also in consideration the game and the optimal strategy to play the game.
Please answer in the following format:
{{"Answer": [ANSWER], "Explanation": <explanation of the answer>}}
Please, respect the format in your answer.
        """
        self.listen(user=user, prompt=final_message)
        self._add_message("user", final_message)
        answer = self.messages[-1]["content"]
        parsed_answer = re.search(r'\{"Answer":\s*(\w+),\s*"Explanation":\s*"(.*?)"\}', answer)
        if parsed_answer:
            answer_value = parsed_answer.group(1)
            explanation_value = parsed_answer.group(2)
            return {"Answer": answer_value, "Explanation": explanation_value}
        else:
            raise ValueError("No valid strategy found in the last response.")