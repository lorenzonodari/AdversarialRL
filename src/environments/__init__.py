import gymnasium as gym
from gymnasium.envs.registration import register as register_env_gym
from ray.tune.registry import register_env as register_env_ray

from .cookie_world import CookieWorld
from .grid_world import GridWorldParams


class PerfectRewardMachine(gym.Wrapper):
    """
    Simple wrapper that allows to enrich an environment with its associated perfect RM.

    The perfect RM is only used by the LRM implementation for debugging purposes and for determining the expected
    reward of an optimal policy.
    """

    def __init__(self, env, perfect_rm):

        super().__init__(env)
        self._perfect_rm = perfect_rm

    def get_perfect_rm(self):
        return self._perfect_rm


class FlattenGridActions(gym.ActionWrapper):
    """
    Flatten a MultiDiscrete([m,n]) bi-dimensional action space - eg: grid coordinates - to a Discrete(m*n) one
    """

    def __init__(self, env):

        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), "Environment must have MultiDiscrete actions"
        assert env.action_space.shape == (2,), "Only 2-dimensional MultiDiscrete spaces are supported"

        super().__init__(env)

        self._n_rows = env.action_space[0].n
        self._n_cols = env.action_space[1].n

        env.action_space = gym.spaces.Discrete(self._n_rows * self._n_cols)

    def action(self, action):

        # Convert action id - ie: the Discrete version of the action - to the proper MultiDiscrete action space
        col = action % self._n_cols
        row = action // self._n_cols

        return row, col


class CookieWorldEnv(gym.Env):
    """
    Gymnasium-compliant implementation of CookieWorld from Icarte et al.

    This class simply wraps the original implementation of the environment in order to comply with Gymnasium APIs.
    NB: Note that this environment has no terminal state: the agent can continue to act indefinitely.
        Nonetheless, the original author assumes a maximum episode length of 5000 steps during its experiments.
        This is achieved, however, by resorting to means that are external to this class.

    """

    def __init__(self, *, seed=None):

        self._seed = seed
        self._params = GridWorldParams('cookieworld', 'maps/cookie.txt', 0.05)  # Movement noise = 5%
        self._world = None
        self._perfect_rm = CookieWorld(self._params).get_perfect_rm()

        self.action_space = gym.spaces.Discrete(4, seed=self._seed)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(256,), seed=self._seed)  # TODO: Fix obs shape

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

        self._world = CookieWorld(self._params, seed=seed)
        obs = self._world._get_map_features()
        info = {
            "events": self._world.get_events(),
            "event_features": self._world._get_event_features()
        }

        return obs, info

    def get_perfect_rm(self):

        return self._perfect_rm


register_env_gym(id='CookieWorld-v0', entry_point="environments:CookieWorldEnv")
register_env_ray("CookieWorld-v0", lambda x: CookieWorldEnv())  # Register env for use with Ray
