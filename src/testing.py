import os
import csv
import json
import time
import argparse

import tensorflow as tf

from lrm.agents import TrainedLRMAgent
from lrm.attacks import gather_traces, preprocess_traces
from lrm.attacks import rank_event_blinding_strategies, rank_edge_blinding_strategies
from lrm.labeling import RandomLFNoise, EventBlindingAttack, EdgeBlindingAttack


def save_results(results, run_time, n_tamperings, session_name, agent_id, *, seed=None):

    results_folder = f'results/test/{session_name}/{agent_id}'

    # Create folder if needed
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    n_episodes, episode_horizon, total_reward, total_steps, traces = results

    # Save test results
    with open(f'{results_folder}/results.csv', 'w', newline='') as results_file:
        writer = csv.writer(results_file)
        writer.writerows([
            ['Episodes', 'Ep. Horizon', 'Total Reward', 'Total Steps', 'Total tamperings', 'Runtime (s)', 'Seed'],
            [n_episodes, episode_horizon, total_reward, total_steps, n_tamperings, run_time, seed]
        ])

    # Save obtained traces
    with open(f'{results_folder}/traces.json', 'w') as traces_file:
        json.dump(traces, traces_file)


def save_blinding_strategies(strategies, session_name, strats_id):

    strategies_folder = f'results/test/{session_name}'

    if not os.path.exists(strategies_folder):
        os.makedirs(strategies_folder)

    with open(f'{strategies_folder}/{strats_id}.csv', 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(['Rank', 'Strategy', 'Avg.Reward', 'Avg.Steps', 'Avg.Tamperings'])

        for i, (strat, scores) in enumerate(strategies):

            avg_reward, avg_steps, avg_tamperings = scores
            writer.writerow([i, str(strat), avg_reward, avg_steps, avg_tamperings])


def test_lf_baseline(agents, n_episodes, episode_horizon, session_name):

    for i, agent_id in enumerate(agents):

        # Load the agent
        agent = TrainedLRMAgent(agent_id)

        # Create environment instance
        env = agent.get_env()

        # Execute test and save results
        start = time.time()
        results = agent.test(env, n_episodes, episode_horizon, seed=i)
        run_time = int(time.time() - start)
        save_results(results, run_time, 0, session_name, agent_id, seed=i)

        agent.close()


def test_lf_random_noise(agents, n_episodes, episode_horizon, session_name):

    for i, agent_id in enumerate(agents):

        # Load the agent
        agent = TrainedLRMAgent(agent_id)

        for noise in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:

            # Prepare the environment + random labeling function noise
            env = RandomLFNoise(agent.get_env(), noise, seed=i)

            # Execute the test and save the results
            start = time.time()
            results = agent.test(env, n_episodes, episode_horizon, seed=i)
            run_time = int(time.time() - start)
            n_tamperings = env.n_tamperings
            save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_noise_{int(noise * 100)}', seed=i)

        agent.close()


def test_event_blinding(agents, traces_session_id, n_strategies, n_episodes, episode_horizon, session_name):

    traces = gather_traces(traces_session_id)
    traces = preprocess_traces(traces)

    for i, agent_id in enumerate(agents):

        # Load the victim agent
        victim = TrainedLRMAgent(agent_id)

        # Test both compound and atomic events
        for compound_events in [True, False]:

            strats_type = 'atom' if not compound_events else 'comp'
            strats_id = f'{agent_id}_{strats_type}'

            ranked_strategies = rank_event_blinding_strategies(agent_id, traces,
                                                               use_compound_events=compound_events,
                                                               trials_per_strategy=100,
                                                               episode_length=500)

            save_blinding_strategies(ranked_strategies, session_name, strats_id)

            # Test only a given number of best strategies
            best_strategies = ranked_strategies[:n_strategies]

            for j, (strat, _) in enumerate(best_strategies):

                target, appearance = strat

                base_env = victim.get_env()
                env = EventBlindingAttack(base_env, target, appearance)

                start = time.time()
                results = victim.test(env, n_episodes, episode_horizon, seed=i)
                run_time = int(time.time() - start)
                n_tamperings = env.n_tamperings
                save_results(results, run_time, n_tamperings, session_name, f'{strats_id}_{j}', seed=i)

        victim.close()


def test_edge_blinding(agents, traces_session_id, n_strategies, n_episodes, episode_horizon, session_name):

    traces = gather_traces(traces_session_id)
    traces = preprocess_traces(traces)

    for i, agent_id in enumerate(agents):

        # Load the victim agent
        victim = TrainedLRMAgent(agent_id)

        # Test both compound and atomic events
        for state_based in [True, False]:

            strats_type = 'edge' if not state_based else 'state'
            strats_id = f'{agent_id}_{strats_type}'

            ranked_strategies = rank_edge_blinding_strategies(agent_id, traces,
                                                              target_states=state_based,
                                                              trials_per_strategy=100,
                                                              episode_length=500)

            save_blinding_strategies(ranked_strategies, session_name, strats_id)

            # Test only a given number of best strategies
            best_strategies = ranked_strategies[:n_strategies]

            for j, (strat, _) in enumerate(best_strategies):

                target, appearance = strat

                base_env = victim.get_env()
                env = EdgeBlindingAttack(base_env, victim._rm, target, appearance)

                start = time.time()
                results = victim.test(env, n_episodes, episode_horizon, seed=i)
                run_time = int(time.time() - start)
                n_tamperings = env.n_tamperings
                save_results(results, run_time, n_tamperings, session_name, f'{strats_id}_{j}', seed=i)

        victim.close()


def test_lrm_agent(cli_args):

    # Gather the agents IDs to be tested
    agents = [agent for agent in os.listdir('agents') if agent.startswith(cli_args.agents_prefix)]

    # Dispatch test execution to the proper function
    if cli_args.test == 'baseline':

        test_lf_baseline(agents, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    elif cli_args.test == 'randomlf':

        test_lf_random_noise(agents, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    elif cli_args.test == 'evt-blind':

        test_event_blinding(agents, cli_args.traces_from, cli_args.n_strategies, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    elif cli_args.test == 'edg-blind':

        test_edge_blinding(agents, cli_args.traces_from, cli_args.n_strategies, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    else:

        print(f'Requested test "{cli_args.test}" not found in the available testing scenarios')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(description='Test pre-trained LRM agents under various conditions')

    args_parser.add_argument('-t', '--test',
                             choices=['baseline', 'randomlf', 'evt-blind', 'edg-blind'],
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

    # Blinding attacks-specific arguments
    args_parser.add_argument('--traces_from',
                             help="[test=*-blind] Name of the session to be used for obtaining the traces to determine attack strategies",
                             required=False)
    args_parser.add_argument('--n_strategies',
                             type=int,
                             help='[test=*-blind] Number of top-rated strategies to be used for actual agent testing',
                             required=False)

    args = args_parser.parse_args()

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place (Or port to tf2 directly)

    test_lrm_agent(args)
