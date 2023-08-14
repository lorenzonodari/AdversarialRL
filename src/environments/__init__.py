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

        # Create one instance of underlying world to get informations
        world = CookieWorld(self._params)
        self._perfect_rm = world.get_perfect_rm()
        self._all_events = world.get_all_events()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(248,), seed=self._seed)

    def step(self, action):

        reward, done = self._world.execute_action(action)

        if self._episodic:
            done = reward == 1

        obs = self._world._get_map_features()

        events = self._world.get_events()
        info = {
            "events": events,
            "event_features": self._world._get_event_features(events)
        }

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._world = CookieWorld(self._params, seed=seed)
        obs = self._world._get_map_features()

        events = self._world.get_events()
        info = {
            "events": events,
            "event_features": self._world._get_event_features(events)
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm

    def get_perfect_rewards(self):

        return {(3, "2C"): 1, (2, '0C'): 1}

    def get_all_events(self):

        return self._all_events

    def get_event_features(self, events):

        return self._world._get_event_features(events)


class KeysWorldEnv(gym.Env):
    """
    Gymnasium-compliant implementation of 2-KeysWorld from Icarte et al.
    """

    def __init__(self, *, seed=None):

        self._seed = seed
        self._params = GridWorldParams('keysworld', 'maps/2-keys.txt', 0.05)  # Movement noise
        self._world = None

        # Create one instance of underlying world to get informations
        world = KeysWorld(self._params)
        self._perfect_rm = world.get_perfect_rm()
        self._all_events = world.get_all_events()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(270,), seed=self._seed)

    def step(self, action):

        reward, done = self._world.execute_action(action)
        obs = self._world._get_map_features()

        events = self._world.get_events()
        info = {
            "events": events,
            "event_features": self._world._get_event_features(events)
        }

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._world = KeysWorld(self._params, seed=seed)
        obs = self._world._get_map_features()

        events = self._world.get_events()
        info = {
            "events": events,
            "event_features": self._world._get_event_features(events)
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm

    def get_perfect_rewards(self):

        return {(6, "3G"): 1}

    def get_all_events(self):

        return self._all_events

    def get_event_features(self, events):

        return self._world._get_event_features(events)


class SymbolWorldEnv(gym.Env):
    """
    Gymnasium-compliant implementation of SymbolWorld from Icarte et al.
    """

    def __init__(self, *, seed=None):

        self._seed = seed
        self._params = GridWorldParams('symbolworld', 'maps/symbol.txt', 0.05)
        self._world = None

        # Create one instance of underlying world to get informations
        world = SymbolWorld(self._params)
        self._perfect_rm = world.get_perfect_rm()
        self._all_events = world.get_all_events()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(512,), seed=self._seed)

    def step(self, action):

        reward, done = self._world.execute_action(action)
        obs = self._world._get_map_features()

        events = self._world.get_events()
        info = {
            "events": events,
            "event_features": self._world._get_event_features(events)
        }

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self._world = SymbolWorld(self._params, seed=seed)
        obs = self._world._get_map_features()

        events = self._world.get_events()
        info = {
            "events": events,
            "event_features": self._world._get_event_features(events)
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm

    def get_perfect_rewards(self):

        return {
            (1, '-1Abc'): 1,
            (1, '-1aBc'): -1,
            (1, '-1abC'): -1,
            (1, '2Abc'): 1,
            (1, '2aBc'): -1,
            (1, '2abC'): -1,

            (2, '-1Abc'): -1,
            (2, '-1aBc'): 1,
            (2, '-1abC'): -1,
            (2, '2Abc'): -1,
            (2, '2aBc'): 1,
            (2, '2abC'): -1,

            (3, '-1Abc'): -1,
            (3, '-1aBc'): -1,
            (3, '-1abC'): 1,
            (3, '2Abc'): -1,
            (3, '2aBc'): -1,
            (3, '2abC'): 1,

            (4, '-1Abc'): 1,
            (4, '-1aBc'): -1,
            (4, '-1abC'): -1,
            (4, '2Abc'): -1,
            (4, '2aBc'): -1,
            (4, '2abC'): -1,

            (5, '-1Abc'): -1,
            (5, '-1aBc'): 1,
            (5, '-1abC'): -1,
            (5, '2Abc'): -1,
            (5, '2aBc'): -1,
            (5, '2abC'): -1,

            (6, '-1Abc'): -1,
            (6, '-1aBc'): -1,
            (6, '-1abC'): 1,
            (6, '2Abc'): -1,
            (6, '2aBc'): -1,
            (6, '2abC'): -1,

            (7, '-1Abc'): -1,
            (7, '-1aBc'): -1,
            (7, '-1abC'): -1,
            (7, '2Abc'): 1,
            (7, '2aBc'): -1,
            (7, '2abC'): -1,

            (8, '-1Abc'): -1,
            (8, '-1aBc'): -1,
            (8, '-1abC'): -1,
            (8, '2Abc'): -1,
            (8, '2aBc'): 1,
            (8, '2abC'): -1,

            (9, '-1Abc'): -1,
            (9, '-1aBc'): -1,
            (9, '-1abC'): -1,
            (9, '2Abc'): -1,
            (9, '2aBc'): -1,
            (9, '2abC'): 1,
        }

    def get_all_events(self):

        return self._all_events

    def get_event_features(self, events):

        return self._world._get_event_features(events)


register_env_gym(id='CookieWorld-v0', entry_point="environments:CookieWorldEnv")
register_env_gym(id='KeysWorld-v0', entry_point="environments:KeysWorldEnv")
register_env_gym(id='SymbolWorld-v0', entry_point="environments:SymbolWorldEnv")
