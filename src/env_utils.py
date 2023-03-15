import gymnasium as gym
import numpy as np

from collections import OrderedDict


class DefineRewards(gym.Wrapper):
    """
    Define a custom reward function for the AICollabWorld-v0 environment.

    NB: This wrapper assumes that the received observations follow the exact format defined
        by the underlying environment (env.unwrapped) observation space. On the other side, it assumes that
        the actions are expressed in the format exposed by PrepareActionsForDQN.
    """

    def __init__(self, env):

        super().__init__(env)
        self._reset_env_info()

    def _reset_env_info(self):
        self._is_moving = False
        self._is_sensing = False
        self._known_items = 0
        self._holding_objects = False

    def reset(self, seed=None, options=None):

        self._reset_env_info()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):

        reward = -0.001  # Time goes on

        # Movement/Actuation action
        if 1 <= action <= 17:

            # Action request while busy
            if self._is_moving:
                reward -= 0.001
            else:
                self._is_moving = True

        # Sensing/Communication action
        elif action >= 18:

            # Action request while busy
            if self._is_sensing:
                reward -= 0.001
            else:
                self._is_sensing = True

                # Check item
                if action >= 21:
                    item_id = action - 21
                    # Requested info about non-existent item
                    if item_id >= self._known_items:
                        reward -= 0.001

        obs, _, terminated, truncated, info = self.env.step(action)

        if obs['objects_held'] == 1:
            reward += 0.020
            self._holding_objects = True

        # Movement action completed succesfully
        if obs['action_status'][0] == 1:
            reward += 0.010
            self._is_moving = False

        # Movement action failed
        elif obs['action_status'][1] == 1:
            reward -= 0.005
            self._is_moving = False

        # Sensing action completed succesfully
        if obs['action_status'][2] == 1:
            reward += 0.001
            self._is_sensing = False

        # Sensing action failed
        elif obs['action_status'][3] == 1:
            reward -= 0.001

        return obs, reward, terminated, truncated, info


class PrepareActionsForDQN(gym.ActionWrapper):
    """
    Prepare the environment for DQN training.

    Stable-Baselines3 implementation of DQN requires Discrete action spaces.
    This wrapper traslates integer IDs that represent actions to their more expressive
    counterpart understood by the underlying environment.

    Moreover, since we want to use single-agent algorithms, we prune the action space in order to
    remove actions that lead to interaction with other agents.
    The resulting action space thus consists of only 21 actions:

     - wait = 0

     # Movement / Actuation
     - move_up = 1
     - move_down = 2
     - move_left = 3
     - move_right = 4
     - move_up_right = 5
     - move_up_left = 6
     - move_down_right = 7
     - move_down_left = 8
     - grab_up = 9
     - grab_right = 10
     - grab_down = 11
     - grab_left = 12
     - grab_up_right = 13
     - grab_up_left = 14
     - grab_down_right = 15
     - grab_down_left = 16
     - drop_object = 17

     # Sensing

     - danger_sensing = 18
     - get_occupancy_map = 19
     - get_objects_held = 20
     - check_item = 21 + (item_id)  e.g: 21 -> check_item(0), 22 -> check_item(1) etc...

    """
    # TODO: Define enum for actions

    def __init__(self, env):

        super().__init__(env)

        self.num_atomic_actions = 21
        self.num_check_items_actions = self.env.action_space['item'].n

        self.action_space = gym.spaces.Discrete(self.num_atomic_actions + self.num_check_items_actions)

    def action(self, action):

        # Convert wait action to actual action_id used by the simulator
        if action == 0:
            return OrderedDict({'action': 26, 'item': 0, 'robot': 0})

        # Handle check_item action
        elif action >= 21:
            action_id = 20
            item_id = action - 21
            return OrderedDict({'action': action_id, 'item': item_id, 'robot': 0})

        # Raise exception on unknown action
        elif action >= self.action_space.n:
            raise NotImplementedError

        else:
            return OrderedDict({'action': action - 1, 'item': 0, 'robot': 0})
