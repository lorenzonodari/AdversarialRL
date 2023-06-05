import os
import csv
import json
import time
import argparse

import tensorflow as tf

from lrm.algorithm import TrainedLRMAgent
from environments import CookieWorldEnv, KeysWorldEnv, SymbolWorldEnv


def get_env_for_agent(agent_id):

    if "_cw_" in agent_id:
        return CookieWorldEnv()
    elif "_kw_" in agent_id:
        return KeysWorldEnv()
    elif "_sw_" in agent_id:
        return SymbolWorldEnv()
    else:
        raise ValueError(f'Unable to detect appropriate env from agent ID "{agent_id}"')


def save_results(results, run_time, session_name, agent_id, *, seed=None):

    results_folder = f'results/test/{session_name}/{agent_id}'

    # Create folder if needed
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    n_episodes, episode_horizon, total_reward, total_steps, traces = results

    # Save test results
    with open(f'{results_folder}/results.csv', 'w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerows([
            ['Episodes', 'Ep. Horizon', 'Total Reward', 'Total Steps', 'Runtime (s)', 'Seed'],
            [n_episodes, episode_horizon, total_reward, total_steps, run_time, seed]
        ])

    # Save obtained traces
    with open(f'{results_folder}/traces.json', 'w') as traces_file:
        json.dump(traces, traces_file)


def test_lf_baseline(agents, n_episodes, episode_horizon, session_name):

    for i, agent_id in enumerate(agents):

        # Load the agent
        agent = TrainedLRMAgent(agent_id)

        # Create environment instance
        env = get_env_for_agent(agent_id)

        # Execute test and save results
        start = time.time()
        results = agent.test(env, n_episodes, episode_horizon, seed=i)
        run_time = int(time.time() - start)
        save_results(results, run_time, session_name, agent_id, seed=i)


def test_lf_random_noise(agents, n_episodes, episode_horizon, session_name, *, noise):

    assert 0.0 < noise < 1.0, "Noise quantity must be in range [0,1]"
    # TODO: Implement
    pass


def test_lf_blinding_attack(agents):

    # TODO
    pass


def test_lrm_agent(cli_args):

    # Gather the agents IDs to be tested
    agents = [agent for agent in os.listdir('agents') if agent.startswith(cli_args.agents_prefix)]

    if cli_args.test == 'baseline':

        test_lf_baseline(agents, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    elif cli_args.test == 'randomlf':

        pass

    else:

        print(f'Requested test "{cli_args.test}" not found in the available testing scenarios')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(description='Test pre-trained LRM agents under various conditions')

    args_parser.add_argument('-t', '--test',
                             choices=['baseline', 'randomlf'],
                             help='The type of test to be run on the agents',  # See test_lrm_agent for the meanings
                             required=True)
    args_parser.add_argument('-n', '--n_episodes',
                             type=int,
                             help='The number of environment episodes to test for',
                             required=True)
    args_parser.add_argument('-m', '--max_episode_length',
                             type=int,
                             help='The maximum number of steps for an episode. Longer episodes are truncated',
                             required=True)
    args_parser.add_argument('-s', '--session',
                             help='The name of this testing session: used to save results',
                             required=True)
    args_parser.add_argument('-a', '--agents_prefix',
                             help='Every agent whose name starts with the given prefix will be tested',
                             required=True)

    # RandomLF-specific arguments
    args_parser.add_argument('--noise',
                             help='[test=randomlf] Noise quantity in range [0,1]',
                             type=float,
                             required=False)

    args = args_parser.parse_args()

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place (Or port to tf2 directly)

    test_lrm_agent(args)
