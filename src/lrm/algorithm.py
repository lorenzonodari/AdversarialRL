import random

import gymnasium as gym
import numpy as np

from environments.game import Game
from .reward_machines.reward_machine import RewardMachine
from .config import LRMConfig
from .dqn import DQN
from .qrm import QRM

"""
- Pseudo-code:
    - Run 'n' random episodes until completion or timeout
    - Learn an RM using those traces
    - Learn a policy for the learned RM until reaching a contradition
    - Add the contradicting trace and relearn the RM
    - Relearn a policy for the new RM from stratch
NOTE:
    - The previous approach can be improved in several ways, but I like its simplicity
    - The policies might be learned using DQN or QRM
"""


def original_lrm_implementation(env: Game, config: LRMConfig, rl='qrm', seed=None):
    """
    Original LRM implementation by Icarte et al.

    This code is maintained mainly as a reference and as a way to test the correctness of our LRM implementation
    when it is modified. Other than this, no use case should see this method as the preferred method of solution, mainly
    due to its incompatibility with Gym environments + lack of useful additions that are included in our implementation.

    :param env: The Game instance representing the environment to use (CookieWorld, KeysWorld or SymbolWorld)
    :param config: The configuration to use for running the algorithm
    :param rl: 'qrm' or 'dqn' depending on which method should be used to learn the policies
    :param seed: The seed to be used for controlling RNGs
    :return: A tuple containing the list of obtained rewards, RM scores and final learned RM
    """

    rm = RewardMachine(config.rm_u_max, config.rm_preprocess, config.rm_tabu_size, config.rm_workers,
                       config.rm_lr_steps,
                       env.get_perfect_rm(), config.use_perfect_rm, seed=seed)
    actions = env.get_actions()
    policy = None
    train_rewards = []
    rm_scores = []
    reward_total = 0
    last_reward = 0
    step = 0

    # Since we discard and recreate policies during execution, plus we don't want to
    # always have the same seed for each of them, we need to make sure that the sequence of seeds
    # is consistent, and depends on the seed given to the algorithm as a whole
    rng = random.Random(seed)
    sub_seed = rng.randint(0, int(4e9))
    sub_seeder = random.Random(sub_seed)

    # Collecting random traces for learning the reward machine
    print("Collecting random traces...")
    while step < config.rm_init_steps:
        # running an episode using a random policy
        env.restart(seed=sub_seeder.randint(0, int(4e9)))
        trace = [(env.get_events(), 0.0)]
        for _ in range(config.episode_horizon):
            # executing a random action
            a = rng.choice(actions)
            reward, done = env.execute_action(a)
            o2_events = env.get_events()
            reward_total += reward
            trace.append((o2_events, reward))
            step += 1
            # Testing
            if step % config.test_freq == 0:
                print("Step: %d\tTrain: %0.1f" % (step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
            # checking if the episode finishes
            if done or config.rm_init_steps <= step:
                if done: rm.add_terminal_observations(o2_events)
                break
                # adding this trace to the set of traces that we use to learn the rm
        rm.add_trace(trace)

    # Learning the reward machine using the collected traces
    print("Learning a reward machines...")
    _, info = rm.learn_the_reward_machine()
    rm_scores.append((step,) + info)

    # Start learning a policy for the current rm
    finish_learning = False
    while step < config.train_steps and not finish_learning:
        env.restart()
        o1_events = env.get_events()
        o1_features = env.get_features()
        u1 = rm.get_initial_state()
        trace = [(o1_events, 0.0)]
        add_trace = False

        for _ in range(config.episode_horizon):

            # reinitializing the policy if the rm changed
            if policy is None:
                print("Learning a policy for the current RM...")
                if rl == "dqn":
                    policy = DQN(config, len(o1_features), len(actions), rm, seed=sub_seeder.randint(0, int(4e9)))
                elif rl == "qrm":
                    policy = QRM(config, len(o1_features), len(actions), rm, seed=sub_seeder.randint(0, int(4e9)))
                else:
                    assert False, "RL approach is not supported yet"

            # selecting an action using epsilon greedy
            a = policy.get_best_action(o1_features, u1, config.epsilon)

            # executing a random action
            reward, done = env.execute_action(a)
            o2_events = env.get_events()
            o2_features = env.get_features()
            u2 = rm.get_next_state(u1, o2_events)

            # updating the number of steps and total reward
            trace.append((o2_events, reward))
            reward_total += reward
            step += 1

            # updating the current RM if needed
            rm.update_rewards(u1, o2_events, reward)
            if done: rm.add_terminal_observations(o2_events)
            if rm.is_observation_impossible(u1, o1_events, o2_events):
                # if o2 is impossible according to the current RM,
                # then the RM has a bug and must be relearned
                add_trace = True

            # Saving this transition
            policy.add_experience(o1_events, o1_features, u1, a, reward, o2_events, o2_features, u2, float(done))

            # Learning and updating the target networks (if needed)
            policy.learn_if_needed()

            # Testing
            if step % config.test_freq == 0:
                print("Step: %d\tTrain: %0.1f" % (step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
                # finishing the experiment if the max number of learning steps was reached
                if policy._get_step() > config.max_learning_steps:
                    finish_learning = True

            # checking if the episode finishes or the agent reaches the maximum number of training steps
            if done or config.train_steps <= step or finish_learning:
                break

                # Moving to the next state
            o1_events, o1_features, u1 = o2_events, o2_features, u2

        # If the trace isn't correctly predicted by the reward machine,
        # we add the trace and relearn the machine
        if add_trace and step < config.train_steps and not finish_learning:
            print("Relearning the reward machine...")
            rm.add_trace(trace)
            same_rm, info = rm.learn_the_reward_machine()
            rm_scores.append((step,) + info)
            if not same_rm:
                # if the RM changed, we have to relearn all the q-values...
                policy.close()
                policy = None
            else:
                print("the new RM is not better than the current RM!!")
                # input()

    if policy is not None:
        policy.close()
        policy = None

    # return the trainig rewards
    return train_rewards, rm_scores, rm.get_info()


class LRMAgent:

    def __init__(self, **kwargs):

        self._config = LRMConfig(**kwargs)
        self._rm = None
        self._policy = None

    def run_lrm(self, env: gym.Env, *, seed=None):
        """
        Implementation of the Learning Reward Machine (LRM) algorithm by Icarte et al.

        This code is based on the original code kindly provided by the authors at:
        https://bitbucket.org/RToroIcarte/lrm/src/master/
        but it is modified in order to support arbitrary gymnasium environments.

        NB: Usable environments must either:

        - implement a labeling function internally and return its output at each step() via the info dict;
        - be wrapped via the LabeledEnv wrapper (which implements the above constraint).

        Moreover, the environment is assumed to have a Discrete action space.

        :param env The gymnasium.Env to be used
        :param seed The seed to be used for execution reproducibility
        :return A tuple containing the list of rewards obtained during training, the list of RM scores and final RM
        """

        assert isinstance(env.action_space, gym.spaces.Discrete), "Only Discrete action spaces are currently supported"

        # Initialization
        self._rm = RewardMachine(self._config.rm_u_max, self._config.rm_preprocess, self._config.rm_tabu_size, self._config.rm_workers,
                                 self._config.rm_lr_steps, env.get_perfect_rm(), self._config.use_perfect_rm,
                                 seed=seed)  # TODO: Fix perfect RM

        self._policy = None
        train_rewards = []
        rm_scores = []
        reward_total = 0
        last_reward = 0
        step = 0

        # Since we discard and recreate policies during execution, plus we don't want to
        # always have the same seed for each of them, we need to make sure that the sequence of seeds
        # is consistent, and depends on the seed given to the algorithm as a whole.
        # Same reasoning applies to the seed given to the env.reset() function.
        sub_seed = random.Random(seed).randint(0, int(4e9))
        sub_seeder = random.Random(sub_seed)

        # Collecting random traces for learning the reward machine
        print("Collecting random traces...")
        while step < self._config.rm_init_steps:

            # Running an episode using a random self._policy
            obs, info = env.reset(seed=sub_seeder.randint(0, int(4e9)))
            trace = [(info["events"], 0.0)]

            for _ in range(self._config.episode_horizon):

                # executing a random action
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                o2_events = info["events"]
                reward_total += reward
                trace.append((o2_events, reward))
                step += 1

                # Testing
                if step % self._config.test_freq == 0:
                    print("Step: %d\tTrain: %0.1f" % (step, reward_total - last_reward))
                    train_rewards.append((step, reward_total - last_reward))
                    last_reward = reward_total

                # Check for episode termination
                if done or self._config.rm_init_steps <= step:
                    if done:
                        self._rm.add_terminal_observations(o2_events)
                    break

            # Add this trace to the set of traces that we use to learn the RM
            self._rm.add_trace(trace)

        # Learn the reward machine using the collected traces
        print("Learning a reward machines...")
        _, info = self._rm.learn_the_reward_machine()
        rm_scores.append((step,) + info)

        # Start learning a self._policy for the current rm
        finish_learning = False
        while step < self._config.train_steps and not finish_learning:

            o1, info = env.reset(seed=sub_seeder.randint(0, int(4e9)))
            o1_features = np.concatenate((o1, info["event_features"]), axis=None)
            o1_events = info["events"]
            u1 = self._rm.get_initial_state()
            trace = [(o1_events, 0.0)]
            add_trace = False

            for _ in range(self._config.episode_horizon):

                # Re-initialize the self._policy if the RM changed
                if self._policy is None:
                    print("Learning a self._policy for the current RM...")
                    if not self._config.use_qrm:
                        self._policy = DQN(self._config, len(o1_features), env.action_space.n, rm, seed=sub_seeder.randint(0, int(4e9)))
                    else:
                        self._policy = QRM(self._config, len(o1_features), env.action_space.n, rm, seed=sub_seeder.randint(0, int(4e9)))

                # Select and execute an action using epsilon greedy
                action = self._policy.get_best_action(o1_features, u1, self._config.epsilon)
                o2, reward, terminated, truncated, info = env.step(action)
                o2_features = np.concatenate((o2, info["event_features"]), axis=None)
                o2_events = info["events"]
                done = terminated or truncated
                u2 = self._rm.get_next_state(u1, o2_events)

                # Update the number of steps and total reward
                trace.append((o2_events, reward))
                reward_total += reward
                step += 1

                # Update the current RM if needed
                self._rm.update_rewards(u1, o2_events, reward)
                if done:
                    self._rm.add_terminal_observations(o2_events)

                # If o2 is impossible according to the current RM we must re-learn it
                if self._rm.is_observation_impossible(u1, o1_events, o2_events):
                    add_trace = True

                # Saving this transition
                self._policy.add_experience(o1_events, o1_features, u1, action, reward, o2_events, o2_features, u2, float(done))

                # Learning and updating the target networks (if needed)
                self._policy.learn_if_needed()

                # Testing
                if step % self._config.test_freq == 0:
                    print("Step: %d\tTrain: %0.1f" % (step, reward_total - last_reward))
                    train_rewards.append((step, reward_total - last_reward))
                    last_reward = reward_total
                    # finishing the experiment if the max number of learning steps was reached
                    if self._policy._get_step() > self._config.max_learning_steps:
                        finish_learning = True

                # Check if the episode finishes or the agent reaches the maximum number of training steps
                if done or self._config.train_steps <= step or finish_learning:
                    break

                    # Move to the next state
                o1_events, o1_features, u1 = o2_events, o2_features, u2

            # If the trace isn't correctly predicted by the reward machine, add the trace and re-learn the machine
            if add_trace and step < self._config.train_steps and not finish_learning:
                print("Relearning the reward machine...")
                self._rm.add_trace(trace)
                same_rm, info = self._rm.learn_the_reward_machine()
                rm_scores.append((step,) + info)

                # If the RM changed, discard the old self._policy to learn a new one
                if not same_rm:
                    self._policy.close()
                    self._policy = None
                else:
                    print("the new RM is not better than the current RM!!")

        if self._policy is not None:
            self._policy.close()
            self._policy = None

        # return the trainig rewards
        return train_rewards, rm_scores, self._rm.get_info()
