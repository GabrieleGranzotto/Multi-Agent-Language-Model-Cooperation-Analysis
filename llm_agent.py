import re, ollama, json, time, random

class OllamaAgent:
    def __init__(self, game, model_name: str = "gemma4"):
        self.model_name = model_name
        self.game = game
        self.messages = []
        self._context_definition()
        self.response_history = []
        self.agent_id = random.randint(1000, 9999)  # Unique identifier for the agent

    def _add_message(self, role: str, content: str):
        # append a message to the conversation history
        # valid roles are "system", "user", and "assistant"
        self.messages.append({"role": role, "content": content})

    def respond(self):
        response = ollama.chat(model = self.model_name, messages = self.messages)
        content = response["message"]["content"]
        self._add_message("assistant", content)
        return content

    def listen(self, prompt: str, user: str = "AGENT"):
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
You are an autonomous agent participating in a behavioral economics experiment, playing a classic coordination game (in this specific context: {self.game.name}).
Your goal is to secure the best possible outcome for yourself. However, to do so, you must carefully weigh the mathematical risks of the payoff matrix and evaluate how much you can actually trust the other agent based on your conversation.
You start chatting with the other agent called [AGENT] about the intention of the game and agree on a strategy to play the game.
Then you will not be able to communicate with the other agent anymore and you will have to play the game.
Another user called [JUDGE] will ask you about your strategy and the game you are playing. 
You will have to answer the questions of the [JUDGE] in a standardized way.
The answer should be in the following format:
Explanation: <explanation of the answer>, Answer: [ANSWER]
The options for the [ANSWER] are mandatory: [{self.game.actions[0]}] or [{self.game.actions[1]}].
The <explanation of the answer> should be a concise explanation of the reasoning behind the chosen answer.
Description of the game: {self.game.description}
        """
        self._add_message("system", context)

    def get_strategy(self):
        """
        Extract the strategy from the agent's last response.
        """
        user = "JUDGE"
        final_message = f"""
[{user}]: Now the other agent can't listen to you. Please provide your final decision.
Take into consideration the previous conversation you had with the other agent, and evaluate whether their words were genuine or deceptive. 
Take also into consideration the mathematical payoffs of the game.
Please answer in the following format:
Explanation: <explanation of the answer>, Answer: [ANSWER]
Please, respect the format in your answer.
        """
        self.listen(user=user, prompt=final_message)
        raw_response = self.respond()
        pattern = r'\s*Explanation\s*:\s*(.*?)\s*,\s*Answer\s*:\s*\[([A-Z_]+)\]\s*'
        match = re.search(pattern, raw_response, re.DOTALL)
        if not match:
            critical_message = "Sorry, your message does not follow the required format. Please provide your answer in the following format:\nExplanation: <explanation of the answer>, Answer: [ANSWER]"
            answer, explanation = self._error_handling(critical_message, user=user, pattern=pattern)
        else:
            answer = match.group(2)
            explanation = match.group(1)
            if answer not in self.game.actions:
                critical_message = f"Sorry, your answer '{answer}' is not a valid option. Please provide your answer in the following format:\nExplanation: <explanation of the answer>, Answer: [ANSWER]\nThe valid options for [ANSWER] are: [{self.game.actions[0]}] or [{self.game.actions[1]}]."
                answer, explanation = self._error_handling(critical_message, user=user, pattern=pattern)
        self._add_to_response_history(answer, explanation)
        return answer, explanation

    def _error_handling(self, critical_message: str, user: str, pattern: str):
        """
        Handle errors in the agent's response by prompting for a valid response.

        Args:
            critical_message (str): The message to display when the agent's response does not match the required format.
            user (str): The user who is providing the response.
            pattern (str): The regex pattern to match the required format.

        Returns:
            tuple: A tuple containing the answer and explanation extracted from the agent's response, or an error message if the response is invalid.
        """
        self.listen(user=user, prompt=critical_message)
        raw_response = self.respond()
        match = re.search(pattern, raw_response, re.DOTALL)
        if not match:
            critical_error_explanation = "The agent failed to provide a valid response in the required format after two attempts."
            return "[ERROR]", critical_error_explanation
        return match.group(2), match.group(1)
    
    def _add_to_response_history(self, answer: str, explanation: str):
        """
        Add the agent's response to the response history.

        Args:
            answer (str): The answer provided by the agent.
            explanation (str): The explanation for the answer.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.response_history.append({"time": timestamp, "id": self.agent_id, "answer": answer, "explanation": explanation})
        self.response_buffer_cnt = len(self.response_history)

        MAX_BUFFER_SIZE = 5  # Define the maximum buffer size for responses
        if self.response_buffer_cnt >= MAX_BUFFER_SIZE:
            self.save_response_history()

    def save_response_history(self):
        """
        Save the response history to a file.
        """
        try:
            with open("response_history.json", "r") as f:
                existing_history = json.load(f)
        except FileNotFoundError:
            existing_history = []
        for response in self.response_history:
            existing_history.append(response)
        with open("response_history.json", "w") as f:
            json.dump(existing_history, f, indent=4)
        self.response_history = []  # Clear the response history after saving