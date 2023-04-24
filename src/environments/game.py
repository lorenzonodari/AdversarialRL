from .keys_world import KeysWorld
from .symbol_world import SymbolWorld
from .cookie_world import CookieWorld

class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """
    def __init__(self, game_params):
        self.game_type   = game_params.game_type
        self.game_params = game_params

class Game:

    def __init__(self, params):
        self.params = params
        self.restart()
        
    def is_env_game_over(self):
        return self.game.env_game_over

    def execute_action(self, action):
        """
        We execute 'action' in the game
        Returns the reward
        """
        return self.game.execute_action(action)

    def restart(self):
        if self.params.game_type == "keysworld":
            self.game = KeysWorld(self.params.game_params)
        if self.params.game_type == "symbolworld":
            self.game = SymbolWorld(self.params.game_params)
        if self.params.game_type == "cookieworld":
            self.game = CookieWorld(self.params.game_params)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.game.get_actions()

    def get_perfect_rm(self):
        """
        Returns a perfect RM for this domain
        """
        return self.game.get_perfect_rm()

    def get_optimal_action(self):
        """
        HACK: returns the best possible action given current state
        """
        return self.game.get_optimal_action()

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.game.get_events()

    def get_all_events(self):
        """
        Returns a string with all the possible events that may occur in the environment
        """
        return self.game.get_all_events()


    def get_state(self):
        """
        Returns a representation of the current state with enough information to 
        compute a reward function using an RM (the format is domain specific)
        """
        return self.game.get_state()

    def get_location(self):
        # this auxiliary method allows to keep track of the agent's movements
        return self.game.get_location()

    def get_features(self):
        return self.game.get_features()
    
    def get_state_and_features(self):
        return self.get_state(), self.get_features()
