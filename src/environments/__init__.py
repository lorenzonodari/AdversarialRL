import gymnasium as gym
from gymnasium.envs.registration import register as register_env_gym
from ray.tune.registry import register_env as register_env_ray

from .cookie_world import CookieWorld
from .grid_world import GridWorldParams


class CookieWorldEnv(gym.Env):
    """Gymnasium-compliant implementation of CookieWorld from Icarte et al.

    This class simply wraps the original implementation of the environment in order to comply with Gymnasium APIs.
    NB: Note that this environment has no terminal state: the agent can continue to act indefinitely.
        Nonetheless, the original author assumes a maximum episode length of 5000 steps during its experiments.
        This is achieved, however, by resorting to means that are external to this class.
    """

    def __init__(self):

        self._params = GridWorldParams('cookie_world', 'maps/cookie.txt', 0.05)  # Movement noise = 5%
        self._world = None

        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(256,))  # NB: Obs include events

    def step(self, action):

        reward, done = self._world.execute_action(action)
        obs = self._world.get_features()

        return obs, reward, done, False, None

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        self._world = CookieWorld(self._params)
        obs = self._world.get_features()

        return obs, {}


register_env_gym(id='CookieWorld-v0', entry_point="environments:CookieWorldEnv")
register_env_ray("CookieWorld-v0", lambda x: CookieWorldEnv())  # Register env for use with Ray
