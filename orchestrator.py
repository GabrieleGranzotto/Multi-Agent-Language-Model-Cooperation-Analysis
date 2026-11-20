import time, json, random
from game_env import GameMatrix
from llm_agent import OllamaAgent

class Orchestrator:
    def __init__(self, max_turns: int = 5, model_name: str = "gemma4", other_model_name: str = None, randomize_start: bool = False, verbose: bool = False):
        self.max_turns = max_turns

        self.model_name = model_name
        self.other_model_name = other_model_name if other_model_name is not None else model_name

        self.rand_start = randomize_start
        self.verbose = verbose
        self.response_history = []
        self.game = GameMatrix.load_from_json("data/games.json", "prisoners_dilemma")
        self.reset()

    def reset(self):
        """
        Reset the system to take in memory just the single conversation.
        The inference should stay light and fast.
        """
        agentA = OllamaAgent(self.game, model_name=self.model_name)
        agentB = OllamaAgent(self.game, model_name=self.other_model_name)
        self.agentA_id = agentA.id
        self.agentB_id = agentB.id
        self.map = {agentA.id: agentA, agentB.id: agentB}
        self.agents = [agentA, agentB]

    def run(self):
        # extract the agent that should start the conversation randomly
        start_agent_idx = random.randint(0, 1) if self.rand_start else 0 #except for the case where randomize_start is False, in which case agentA always starts
        first_agent = self.agents[start_agent_idx]
        other_agent = self.agents[1 - start_agent_idx]
        # start the talking phase between the two agents
        self._talking_phase(first_agent, other_agent)
        # after the talking phase, ask each agent for their final strategy
        strategyA, explA = self.agents[0].get_strategy()
        strategyB, explB = self.agents[1].get_strategy()
        try:
            payoffA, payoffB = self.game.resolve_turn(strategyA, strategyB)
        except ValueError:
            payoffA, payoffB = 0, 0
        self.result_data = [(strategyA, strategyB), (payoffA, payoffB), (explA, explB), first_agent.id]
        self._add_to_response_history()

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
                print(f"[{self.map[agent.id]}]: {message}") if self.verbose else None

    def result(self):
        strategyA, strategyB = self.result_data[0]
        payoffA, payoffB = self.result_data[1]
        explA, explB = self.result_data[2]
        print(f"Agent A chose: {strategyA} \nExplanation: {explA}\n\n")
        print(f"Agent B chose: {strategyB} \nExplanation: {explB}\n\n")
        print(f"Game resolved. \nPayoffs:")
        print(f"Agent A: {payoffA}, \nAgent B: {payoffB}")

    def _add_to_response_history(self):
        """
        Add the agent's response to the response history.
        """
        strategyA, strategyB = self.result_data[0]
        payoffA, payoffB = self.result_data[1]
        explA, explB = self.result_data[2]
        first_agent_id = self.result_data[3]
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
            "first_speaker_id": first_agent_id,
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
        path = "data/response_history.json"
        try:
            with open(path, "r") as f:
                existing_history = json.load(f)
        except FileNotFoundError:
            existing_history = []
        for response in self.response_history:
            existing_history.append(response)
        with open(path, "w") as f:
            json.dump(existing_history, f, indent=4)
        self.response_history = []  # Clear the response history after saving