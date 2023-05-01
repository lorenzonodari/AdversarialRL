import random
import time

import gymnasium as gym
import numpy as np

from ..reward_machines.reward_machine import RewardMachine
from .learning_parameters import LearningParameters
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


def get_default_lrm_config():

    lp = LearningParameters()

    lp.set_test_parameters(test_freq=int(1e4))

    lp.set_rm_learning(rm_init_steps=200e3,
                       rm_u_max=10,
                       rm_preprocess=True,
                       rm_tabu_size=10000,
                       rm_lr_steps=10,
                       rm_workers=8)

    lp.set_rl_parameters(gamma=0.9,
                         train_steps=int(3e6),
                         episode_horizon=int(5e3),
                         epsilon=0.1,
                         max_learning_steps=int(3e6))

    lp.set_deep_rl(lr=5e-5,
                   learning_starts=50000,
                   train_freq=1,
                   target_network_update_freq=100,
                   buffer_size=100000,
                   batch_size=32,
                   use_double_dqn=True,
                   num_hidden_layers=5,
                   num_neurons=64,
                   use_qrm=True)

    return lp


def original_run_lrm(env_params, lp, rl='qrm'):

    from environments.game import Game

    env = Game(env_params)
    rm = RewardMachine(lp.rm_u_max, lp.rm_preprocess, lp.rm_tabu_size, lp.rm_workers, lp.rm_lr_steps,
                       env.get_perfect_rm(), lp.use_perfect_rm)
    actions = env.get_actions()
    policy = None
    train_rewards = []
    rm_scores = []
    reward_total = 0
    last_reward = 0
    step = 0

    # Collecting random traces for learning the reward machine
    print("Collecting random traces...")
    while step < lp.rm_init_steps:
        # running an episode using a random policy
        env.restart()
        trace = [(env.get_events(), 0.0)]
        for _ in range(lp.episode_horizon):
            # executing a random action
            a = random.choice(actions)
            reward, done = env.execute_action(a)
            o2_events = env.get_events()
            reward_total += reward
            trace.append((o2_events, reward))
            step += 1
            # Testing
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f" % (step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
            # checking if the episode finishes
            if done or lp.rm_init_steps <= step:
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
    while step < lp.train_steps and not finish_learning:
        env.restart()
        o1_events = env.get_events()
        o1_features = env.get_features()
        u1 = rm.get_initial_state()
        trace = [(o1_events, 0.0)]
        add_trace = False

        for _ in range(lp.episode_horizon):

            # reinitializing the policy if the rm changed
            if policy is None:
                print("Learning a policy for the current RM...")
                if rl == "dqn":
                    policy = DQN(lp, len(o1_features), len(actions), rm)
                elif rl == "qrm":
                    policy = QRM(lp, len(o1_features), len(actions), rm)
                else:
                    assert False, "RL approach is not supported yet"

            # selecting an action using epsilon greedy
            a = policy.get_best_action(o1_features, u1, lp.epsilon)

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
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f" % (step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
                # finishing the experiment if the max number of learning steps was reached
                if policy._get_step() > lp.max_learning_steps:
                    finish_learning = True

            # checking if the episode finishes or the agent reaches the maximum number of training steps
            if done or lp.train_steps <= step or finish_learning:
                break

                # Moving to the next state
            o1_events, o1_features, u1 = o2_events, o2_features, u2

        # If the trace isn't correctly predicted by the reward machine,
        # we add the trace and relearn the machine
        if add_trace and step < lp.train_steps and not finish_learning:
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


def run_lrm(env: gym.Env, lp: LearningParameters = None):
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
    :param lp The LearningParameters instance containing the desired algorithm configuration
    :return The reward sequence obtained during training
    """

    assert isinstance(env.action_space, gym.spaces.Discrete), "Only Discrete action spaces are currently supported"

    if lp is None:
        lp = get_default_lrm_config()

    # Initialization
    rm = RewardMachine(lp.rm_u_max, lp.rm_preprocess, lp.rm_tabu_size, lp.rm_workers, lp.rm_lr_steps, env.get_perfect_rm(), lp.use_perfect_rm)  # TODO: Fix perfect RM

    policy = None
    train_rewards = []
    rm_scores = []
    reward_total = 0
    last_reward = 0
    step = 0

    # Collecting random traces for learning the reward machine
    print("Collecting random traces...")
    while step < lp.rm_init_steps:

        # Running an episode using a random policy
        obs, info = env.reset()
        trace = [(info["events"], 0.0)]

        for _ in range(lp.episode_horizon):

            # executing a random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            o2_events = info["events"]
            reward_total += reward
            trace.append((o2_events, reward))
            step += 1

            # Testing
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total

            # Check for episode termination
            if done or lp.rm_init_steps <= step:
                if done:
                    rm.add_terminal_observations(o2_events)
                break

        # Add this trace to the set of traces that we use to learn the RM
        rm.add_trace(trace)

    # Learn the reward machine using the collected traces
    print("Learning a reward machines...")
    _, info = rm.learn_the_reward_machine()
    rm_scores.append((step,) + info)

    # Start learning a policy for the current rm
    finish_learning = False
    while step < lp.train_steps and not finish_learning:

        o1, info = env.reset()
        o1_features = np.concatenate((o1, info["event_features"]), axis=None)
        o1_events = info["events"]
        u1 = rm.get_initial_state()
        trace = [(o1_events, 0.0)]
        add_trace = False
        
        for _ in range(lp.episode_horizon):

            # Re-initialize the policy if the RM changed
            if policy is None:
                print("Learning a policy for the current RM...")
                if not lp.use_qrm:
                    policy = DQN(lp, len(o1_features), env.action_space.n, rm)
                else:
                    policy = QRM(lp, len(o1_features), env.action_space.n, rm)

            # Select and execute an action using epsilon greedy
            action = policy.get_best_action(o1_features, u1, lp.epsilon)
            o2, reward, terminated, truncated, info = env.step(action)
            o2_features = np.concatenate((o2, info["event_features"]), axis=None)
            o2_events = info["events"]
            done = terminated or truncated
            u2 = rm.get_next_state(u1, o2_events)

            # Update the number of steps and total reward
            trace.append((o2_events, reward))
            reward_total += reward
            step += 1

            # Update the current RM if needed
            rm.update_rewards(u1, o2_events, reward)
            if done:
                rm.add_terminal_observations(o2_events)

            # If o2 is impossible according to the current RM we must re-learn it
            if rm.is_observation_impossible(u1, o1_events, o2_events):
                add_trace = True

            # Saving this transition
            policy.add_experience(o1_events, o1_features, u1, action, reward, o2_events, o2_features, u2, float(done))

            # Learning and updating the target networks (if needed)
            policy.learn_if_needed()

            # Testing
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
                # finishing the experiment if the max number of learning steps was reached
                if policy._get_step() > lp.max_learning_steps:
                    finish_learning = True

            # Check if the episode finishes or the agent reaches the maximum number of training steps
            if done or lp.train_steps <= step or finish_learning: 
                break 

            # Move to the next state
            o1_events, o1_features, u1 = o2_events, o2_features, u2

        # If the trace isn't correctly predicted by the reward machine, add the trace and re-learn the machine
        if add_trace and step < lp.train_steps and not finish_learning:
            print("Relearning the reward machine...")
            rm.add_trace(trace)
            same_rm, info = rm.learn_the_reward_machine()
            rm_scores.append((step,) + info)

            # If the RM changed, discard the old policy to learn a new one
            if not same_rm:
                policy.close()
                policy = None
            else:
                print("the new RM is not better than the current RM!!")

    if policy is not None:
        policy.close()
        policy = None

    # return the trainig rewards
    return train_rewards, rm_scores, rm.get_info()


# def run_lrm_experiments(env_params, lp, rl, n_seed, save):
#
#     time_init = time.time()
#     random.seed(n_seed)
#     rewards, scores, rm_info = run_lrm(env_params, lp, rl)
#     if save:
#         # Saving the results
#         out_folder = "LRM/" + rl + "/" + env_params.game_type
#         save_results(rewards, scores, rm_info, out_folder, 'lrm', rl, n_seed)
#
#     # Showing results
#     print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins\n")
