import gymnasium as gym


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
