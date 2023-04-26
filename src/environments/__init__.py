import gymnasium as gym
from gymnasium.envs.registration import register as register_env_gym
from ray.tune.registry import register_env as register_env_ray

from .cookie_world import CookieWorld
from .grid_world import GridWorldParams


class LabeledEnvironment(gym.Wrapper):
    """
    Wrapper that enriches an environment with a Labeling Function

    The Labeling Function is a function from environment observations to propositional symbols that represent
    high-level events that can be known to have verified in the environment given the last observation received.

    """

    def __init__(self, env, labeling_function):

        super().__init__(env)
        self._labfun = labeling_function

    def step(self, action):
        """
        Step the environment and compute the labeling function for the returned observation.

        In order not to interfere with the environment observation space, events generated via the labeling function
        are returned by the step() function via the info dictionary, at the "events" key.
        The original environment is thus required not to return any information itself at the said key.

        :param action: The action to take in the environment
        :return: The obs, reward, terminated, truncated, info tuple, with info["events"] containing the labeling
                 function output, ie: info["events"] = labeling_function(obs)
        """

        obs, reward, terminated, truncated, info = self.env.step(action)
        info['events'] = self._labfun(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):

        self.env.reset(seed=seed, options=options)

    def close(self):

        self.env.close()



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

        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        self._world = CookieWorld(self._params)
        obs = self._world.get_features()

        return obs, {}


register_env_gym(id='CookieWorld-v0', entry_point="environments:CookieWorldEnv")
register_env_ray("CookieWorld-v0", lambda x: CookieWorldEnv())  # Register env for use with Ray
