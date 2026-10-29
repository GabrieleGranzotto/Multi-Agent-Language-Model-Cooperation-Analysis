class GameMatrix:
    def __init__(self, name:str, description: str, actions: list, payoff_matrix: dict):
        self.name = name # the name of the game
        self.description = description # a brief description of the game
        self.actions = actions # the list of actions available to each player
        self.payoff_matrix = payoff_matrix # given actions, returns the payoff for each player

    @classmethod
    def load_from_json(cls, file_path: str, game_id: str):
        """
        Load a game matrix from a JSON file.
        """
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if game_id not in data:
            raise ValueError(f"Game ID '{game_id}' not found in the JSON file.")

        game_data = data[game_id]

        parsed_payoffs = {}
        for act_A, outcomes in game_data["payoffs"].items():
            for act_B, scores in outcomes.items():
                parsed_payoffs[(act_A, act_B)] = tuple(scores)

        return cls(
            name=game_data['name'],
            description=game_data['description'],
            actions=game_data['actions'],
            payoff_matrix=parsed_payoffs
        )

    def resolve_turn(self, action1, action2):
        """
        Given the actions of both players, return the payoffs for each player.
        """
        if action1 not in self.actions or action2 not in self.actions:
            raise ValueError("Invalid actions provided.")
        
        return self.payoff_matrix[(action1, action2)]
