import os
import csv
import json
import time
import argparse

import tensorflow as tf

from lrm.agents import TrainedLRMAgent
from lrm.attacks import gather_traces, preprocess_traces
from lrm.attacks import simple_event_blinding_strategies, simple_edge_blinding_strategies
from lrm.attacks import ranked_event_blinding_strategies, ranked_edge_blinding_strategies
from lrm.labeling import RandomLFNoise, RandomBlinding, EventBlindingAttack, EdgeBlindingAttack


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


def save_ranked_blinding_strategies(strategies, session_name, strats_id):

    strategies_folder = f'results/test/{session_name}'

    if not os.path.exists(strategies_folder):
        os.makedirs(strategies_folder)

    with open(f'{strategies_folder}/{strats_id}_strategies.csv', 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(['Rank', 'Strategy', 'Avg.Reward', 'Avg.Steps', 'Avg.Tamperings'])

        for i, (strat, scores) in enumerate(strategies):

            avg_reward, avg_steps, avg_tamperings = scores
            writer.writerow([i, str(strat), avg_reward, avg_steps, avg_tamperings])


def save_simple_blinding_strategies(strategies, session_name, strats_id):

    strategies_folder = f'results/test/{session_name}'

    if not os.path.exists(strategies_folder):
        os.makedirs(strategies_folder)

    with open(f'{strategies_folder}/{strats_id}_strategies.csv', 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(['Strategy'])

        for strat in strategies:
            writer.writerow([str(strat)])


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


def test_random_blinding(agents, n_episodes, episode_horizon, session_name):

    for i, agent_id in enumerate(agents):

        # Load the agent
        agent = TrainedLRMAgent(agent_id)

        for noise in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:

            # Prepare the environment + random labeling function noise
            env = RandomBlinding(agent.get_env(), noise, seed=i)

            # Execute the test and save the results
            start = time.time()
            results = agent.test(env, n_episodes, episode_horizon, seed=i)
            run_time = int(time.time() - start)
            n_tamperings = env.n_tamperings
            save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_noise_{int(noise * 100)}', seed=i)

        agent.close()


def test_simple_event_blinding(agents, traces_session_id, n_episodes, episode_horizon, session_name):

    traces = gather_traces(traces_session_id)
    traces = preprocess_traces(traces)

    # Test both compound and atomic events
    for compound_events in [True, False]:
        
        strats_type = 'atom' if not compound_events else 'comp'

        strategies = simple_event_blinding_strategies(traces, use_compound_events=compound_events)
        save_simple_blinding_strategies(strategies, session_name, f'simple_{strats_type}')

        for i, agent_id in enumerate(agents):

            victim = TrainedLRMAgent(agent_id)

            timed_strat, persistent_strat, triggered_strat = strategies

            # Test timed strategy
            strats_id = f'simple_{strats_type}_first'

            base_env = victim.get_env()
            env = EventBlindingAttack(base_env, timed_strat[0], timed_strat[1])

            start = time.time()
            results = victim.test(env, n_episodes, episode_horizon, seed=i)
            run_time = int(time.time() - start)
            n_tamperings = env.n_tamperings
            save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_{strats_id}', seed=i)

            # Test persistent strategy
            strats_id = f'simple_{strats_type}_all'

            base_env = victim.get_env()
            env = EventBlindingAttack(base_env, persistent_strat[0], persistent_strat[1])

            start = time.time()
            results = victim.test(env, n_episodes, episode_horizon, seed=i)
            run_time = int(time.time() - start)
            n_tamperings = env.n_tamperings
            save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_{strats_id}', seed=i)

            # Test triggered strategy with different trigger probabilities
            for trigger_chance in [0.3, 0.4, 0.5]:

                strats_id = f'simple_{strats_type}_trigger_{int(trigger_chance * 100)}'

                base_env = victim.get_env()
                env = EventBlindingAttack(base_env, triggered_strat[0], triggered_strat[1], trigger_chance=trigger_chance, seed=i)

                start = time.time()
                results = victim.test(env, n_episodes, episode_horizon, seed=i)
                run_time = int(time.time() - start)
                n_tamperings = env.n_tamperings
                save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_{strats_id}', seed=i)

            victim.close()


def test_ranked_event_blinding(agents, traces_session_id, n_strategies, n_episodes, episode_horizon, session_name):

    traces = gather_traces(traces_session_id)
    traces = preprocess_traces(traces)

    for i, agent_id in enumerate(agents):

        # Load the victim agent
        victim = TrainedLRMAgent(agent_id)

        # Test both compound and atomic events
        for compound_events in [True, False]:

            strats_type = 'atom' if not compound_events else 'comp'
            strats_id = f'{agent_id}_{strats_type}'

            ranked_strategies = ranked_event_blinding_strategies(agent_id, traces,
                                                                 use_compound_events=compound_events,
                                                                 trials_per_strategy=100,
                                                                 episode_length=500)

            save_ranked_blinding_strategies(ranked_strategies, session_name, strats_id)

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


def test_simple_edge_blinding(agents, traces_session_id, n_episodes, episode_horizon, session_name):

    traces = gather_traces(traces_session_id)
    traces = preprocess_traces(traces)

    # Test both edge-blinding and state-blinding
    for target_states in [True, False]:

        strats_type = 'edge' if not target_states else 'state'

        strategies = simple_edge_blinding_strategies(traces, target_states=target_states)
        save_simple_blinding_strategies(strategies, session_name, f'simple_{strats_type}')

        for i, agent_id in enumerate(agents):

            victim = TrainedLRMAgent(agent_id)

            timed_strat, persistent_strat, triggered_strat = strategies

            # Test timed strategy
            strats_id = f'simple_{strats_type}_first'

            base_env = victim.get_env()
            env = EdgeBlindingAttack(base_env, victim._rm, timed_strat[0], timed_strat[1])

            start = time.time()
            results = victim.test(env, n_episodes, episode_horizon, seed=i)
            run_time = int(time.time() - start)
            n_tamperings = env.n_tamperings
            save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_{strats_id}', seed=i)

            # Test persistent strategy
            strats_id = f'simple_{strats_type}_all'

            base_env = victim.get_env()
            env = EdgeBlindingAttack(base_env, victim._rm, persistent_strat[0], persistent_strat[1])

            start = time.time()
            results = victim.test(env, n_episodes, episode_horizon, seed=i)
            run_time = int(time.time() - start)
            n_tamperings = env.n_tamperings
            save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_{strats_id}', seed=i)

            # Test triggered strategy with different trigger probabilities
            for trigger_chance in [0.3, 0.4, 0.5]:
                strats_id = f'simple_{strats_type}_trigger_{int(trigger_chance * 100)}'

                base_env = victim.get_env()
                env = EdgeBlindingAttack(base_env, victim._rm, triggered_strat[0], triggered_strat[1],
                                         trigger_chance=trigger_chance, seed=i)

                start = time.time()
                results = victim.test(env, n_episodes, episode_horizon, seed=i)
                run_time = int(time.time() - start)
                n_tamperings = env.n_tamperings
                save_results(results, run_time, n_tamperings, session_name, f'{agent_id}_{strats_id}', seed=i)

            victim.close()


def test_ranked_edge_blinding(agents, traces_session_id, n_strategies, n_episodes, episode_horizon, session_name):

    traces = gather_traces(traces_session_id)
    traces = preprocess_traces(traces)

    for i, agent_id in enumerate(agents):

        # Load the victim agent
        victim = TrainedLRMAgent(agent_id)

        # Test both compound and atomic events
        for state_based in [True, False]:

            strats_type = 'edge' if not state_based else 'state'
            strats_id = f'{agent_id}_{strats_type}'

            ranked_strategies = ranked_edge_blinding_strategies(agent_id, traces,
                                                                target_states=state_based,
                                                                trials_per_strategy=100,
                                                                episode_length=500)

            save_ranked_blinding_strategies(ranked_strategies, session_name, strats_id)

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

    elif cli_args.test == 'rand-blind':

        test_random_blinding(agents, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    elif cli_args.test == 'evt-blind':

        if cli_args.rank_strategies:
            test_ranked_event_blinding(agents, cli_args.traces_from, cli_args.n_strategies, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)
        else:
            test_simple_event_blinding(agents, cli_args.traces_from, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    elif cli_args.test == 'edg-blind':

        if cli_args.rank_strategies:
            test_ranked_edge_blinding(agents, cli_args.traces_from, cli_args.n_strategies, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)
        else:
            test_simple_edge_blinding(agents, cli_args.traces_from, cli_args.n_episodes, cli_args.max_episode_length, cli_args.session)

    else:

        print(f'Requested test "{cli_args.test}" not found in the available testing scenarios')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(description='Test pre-trained LRM agents under various conditions')

    args_parser.add_argument('-t', '--test',
                             choices=['baseline', 'randomlf', 'rand-blind', 'evt-blind', 'edg-blind'],
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
                             help="[test=<evt,edg>-blind] Name of the session to be used for obtaining the traces to determine attack strategies",
                             required=False)
    args_parser.add_argument('--n_strategies',
                             type=int,
                             help='[test=<evt,edg>-blind and rank_strategies=True] Number of top-ranked strategies to be used for actual agent testing',
                             required=False)
    args_parser.add_argument('--rank_strategies',
                             help='[test=<evt,edg>-blind] If given, use strategy-ranking variation of the test',
                             action='store_true',
                             required=False)

    args = args_parser.parse_args()

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place (Or port to tf2 directly)

    test_lrm_agent(args)
