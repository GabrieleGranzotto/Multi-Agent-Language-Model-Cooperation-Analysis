import time, json
from game_env import GameMatrix
from llm_agent import OllamaAgent
import numpy as np

class Orchestrator:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.response_history = []
        self.game = GameMatrix.load_from_json("data/games.json", "prisoners_dilemma")
        agentA = OllamaAgent(self.game, model_name="gemma4")
        agentB = OllamaAgent(self.game, model_name="gemma4")
        self.agentA_id = agentA.agent_id
        self.agentB_id = agentB.agent_id
        self.agents = [agentA, agentB]

    def run(self):
        # extract the agent that should start the conversation randomly
        start_agent_idx = np.random.randint(0, 2)
        first_agent = self.agents[start_agent_idx]
        other_agent = self.agents[1 - start_agent_idx]
        # start the talking phase between the two agents
        self._talking_phase(first_agent, other_agent)
        # after the talking phase, ask each agent for their final strategy
        strategyA, explA = self.agents[0].get_strategy()
        strategyB, explB = self.agents[1].get_strategy()
        payoffA, payoffB = self.game.resolve_turn(strategyA, strategyB)
        self.result_data = [(strategyA, strategyB), (payoffA, payoffB), (explA, explB)]
        self._add_to_response_history()
        self.result()

    def _talking_phase(self, first_agent: OllamaAgent, other_agent: OllamaAgent):
        message = "Starting the conversation. Please discuss your strategy for the game."
        user = "JUDGE"

        agents = [other_agent, first_agent]
        first_agent.listen(message, user)
        message = first_agent.respond()

        for _ in range(self.max_turns):
            for agent in agents:
                agent.listen(message, user="AGENT")
                message = agent.respond()

    def result(self):
        strategyA, strategyB = self.result_data[0]
        payoffA, payoffB = self.result_data[1]
        explA, explB = self.result_data[2]
        print(f"Agent A chose: {strategyA}, Explanation: {explA}")
        print(f"Agent B chose: {strategyB}, Explanation: {explB}")
        print(f"Game resolved. Payoffs:")
        print(f"Agent A: {payoffA}, Agent B: {payoffB}")

    def _add_to_response_history(self):
        """
        Add the agent's response to the response history.
        """
        strategyA, strategyB = self.result_data[0]
        payoffA, payoffB = self.result_data[1]
        explA, explB = self.result_data[2]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "time": timestamp,
            "game": self.game.name,
            "agentA": {
                "id": self.agentA_id,
                "answer": strategyA,
                "explanation": explA
            },
            "agentB": {
                "id": self.agentB_id,
                "answer": strategyB,
                "explanation": explB
            },
            "results": {
                "payoffA": payoffA,
                "payoffB": payoffB
            },
            "full_conversation": self.agents[0].messages
        }
        self.response_history.append(record)
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