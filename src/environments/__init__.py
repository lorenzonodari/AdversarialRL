import gymnasium as gym
from gymnasium.envs.registration import register as register_env_gym

from .cookie_world import CookieWorld
from .keys_world import KeysWorld
from .symbol_world import SymbolWorld
from .grid_world import GridWorldParams


class CookieWorldEnv(gym.Env):
    """
    Gymnasium-compliant implementation of CookieWorld from Icarte et al.

    This class simply wraps the original implementation of the environment in order to comply with Gymnasium APIs.
    NB: Note that this environment has no terminal state: the agent can continue to act indefinitely.
        Nonetheless, the original author assumes a maximum episode length of 5000 steps during its experiments.
        This is achieved, however, by resorting to means that are external to this class.

    """

    def __init__(self, *, episodic=True, seed=None):
        """
        Initialize an instance of the CookieWorld environment.

        :param episodic: If True, each episode will terminate when the agent reaches the cookie.
        :param seed: The seed to be used to initialize RNG sources
        """

        self._episodic = episodic
        self._seed = seed
        self._params = GridWorldParams('cookieworld', 'maps/cookie.txt', 0.05)  # Movement noise = 5%
        self._world = None
        self._perfect_rm = CookieWorld(self._params).get_perfect_rm()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(248,), seed=self._seed)

    def step(self, action):

        reward, done = self._world.execute_action(action)

        if self._episodic:
            done = reward == 1

        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._world = CookieWorld(self._params, seed=seed)
        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm


class KeysWorldEnv(gym.Env):
    """
    Gymnasium-compliant implementation of 2-KeysWorld from Icarte et al.

    This class simply wraps the original implementation of the environment in order to comply with Gymnasium APIs.
    NB: Note that this environment has no terminal state: the agent can continue to act indefinitely.
    Nonetheless, the original author assumes a maximum episode length of 5000 steps during its experiments.
    This is achieved, however, by resorting to means that are external to this class.

    """

    def __init__(self, *, seed=None):

        self._seed = seed
        self._params = GridWorldParams('keysworld', 'maps/2-keys.txt', 0.05)  # Movement noise
        self._world = None
        self._perfect_rm = KeysWorld(self._params).get_perfect_rm()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(270,), seed=self._seed)

    def step(self, action):

        reward, done = self._world.execute_action(action)
        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._world = KeysWorld(self._params, seed=seed)
        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm


class SymbolWorldEnv(gym.Env):

    def __init__(self, *, seed=None):

        self._seed = seed
        self._params = GridWorldParams('symbolworld', 'maps/symbol.txt', 0.05)
        self._world = None
        self._perfect_rm = SymbolWorld(self._params).get_perfect_rm()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(512,), seed=self._seed)

    def step(self, action):

        reward, done = self._world.execute_action(action)
        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._world = SymbolWorld(self._params, seed=seed)
        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm


register_env_gym(id='CookieWorld-v0', entry_point="environments:CookieWorldEnv")
register_env_gym(id='KeysWorld-v0', entry_point="environments:KeysWorldEnv")
register_env_gym(id='SymbolWorld-v0', entry_point="environments:SymbolWorldEnv")
