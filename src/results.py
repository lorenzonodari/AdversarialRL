from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import json

from typing import Union


def get_precentiles(a):
    p25 = float(np.percentile(a, 25))
    p50 = float(np.percentile(a, 50))
    p75 = float(np.percentile(a, 75))
    return p25, p50, p75


def format_execution_time(exec_seconds):

    exec_mins = exec_seconds // 60
    remaining_secs = exec_seconds % 60

    if exec_mins < 60:
        return f'{exec_mins} m, {remaining_secs} s'

    exec_hours = exec_mins // 60
    remaining_mins = exec_mins % 60

    if exec_hours < 24:
        return f'{exec_hours} h, {remaining_mins} m'

    exec_days = exec_hours // 24
    remaining_hours = exec_hours % 24

    return f'{exec_days} d, {remaining_hours} h'


def process_training_results(session_folder, execution):
    """
    Process the results obtained from a single training execution.

    :param session_folder: The folder containing the results of the session for this execution
    :param execution: The id of the given execution e.g: seed_42
    :return rewards, steps, runtime = list of obtained rewards, list of step number for each test, execution time in seconds
    """

    results_folder = os.path.join(session_folder, execution)

    # Read execution time
    with open(f'{results_folder}/execution_time.txt') as exec_time_file:
        execution_time_s = int(exec_time_file.readline().strip())

    exec_time_str = format_execution_time(execution_time_s)

    # Reward vs steps plot
    with open(f'{results_folder}/rewards.csv') as rewards_file:
        rewards, steps = [], []
        reader = csv.reader(rewards_file)
        next(reader)  # Discard CSV header
        for row in reader:
            step, reward = row
            steps.append(int(step))
            rewards.append(float(reward))

    plt.figure()
    seed = int(execution.strip('seed_'))
    plt.suptitle(f'[Session: {session_folder} - Seed: {seed}]')
    plt.title(f'Execution time: {exec_time_str}')
    plt.plot(steps, rewards)
    plt.ylabel('Reward')
    plt.xlabel('Training steps')

    os.makedirs(f'{results_folder}/plots', exist_ok=True)
    plt.savefig(f'{results_folder}/plots/rewards_plot.png')
    plt.close()

    return rewards, steps, execution_time_s


def training_session_results(sessions: Union[list, str]) -> None:
    """
    Process the results for every training execution in the given session(s).

    This function produces:

    - a plot of obtained reward during a random episode vs number of training steps;
    - a summary plot of reward vs number of training steps obtained by averaging every run in the session;

    :param sessions: The name of a training session, or a list of them, for which to process results
    """

    if isinstance(sessions, str):
        sessions = [sessions]

    for session in sessions:

        all_rewards = []
        all_times = []
        steps = None

        session_folder = f'results/train/{session}'
        for run_folder in [f for f in os.listdir(session_folder) if f.startswith("seed_")]:

            rewards, current_steps, exec_time = process_training_results(session_folder, run_folder)
            all_rewards.append(rewards)
            all_times.append(exec_time)

            if steps is None:
                steps = current_steps
            else:
                assert steps == current_steps, 'All runs in a single session should share the same test frequency'

        # Generate cumulative results by averaging
        mean_rewards = np.mean(all_rewards, axis=0)
        mean_time = int(np.sum(all_times) / len(all_times))
        mean_time_str = format_execution_time(mean_time)

        # Plot summary
        plt.figure()
        plt.suptitle(f'[Session: {session} - Summary of {len(all_times)} runs]')
        plt.title(f'Avg. execution time: {mean_time_str}')
        plt.plot(steps, mean_rewards)
        plt.xlabel('Training steps')
        plt.ylabel('Avg. reward')
        plt.savefig(f'{session_folder}/summary_plot.png')
        plt.close()


def process_test_results(session_folder, agent_id):

    # Read results.csv file for general info and totals
    with open(f'{session_folder}/{agent_id}/results.csv', newline='') as results_file:
        reader = csv.reader(results_file)

        # Discard header
        _ = next(reader)

        # Get info
        n_episodes, episode_horizon, tot_reward, tot_steps, tot_tamperings, _, _ = next(reader)

    # Convert to proper data types
    n_episodes = int(n_episodes)
    episode_horizon = int(episode_horizon)
    tot_reward = int(tot_reward)
    tot_steps = int(tot_steps)
    tot_tamperings = int(tot_tamperings)

    n_success, n_failures = 0, 0
    success_steps, failure_steps = 0, 0
    success_reward, failure_reward = 0, 0

    # Read traces.json for fine-grained info
    with open(f'{session_folder}/{agent_id}/traces.json') as traces_file:

        traces = json.load(traces_file)

        assert len(traces) == n_episodes, "Content of traces.json is incoherent with results.csv"

        for _, (reward, steps) in traces:

            if reward == 1:
                n_success += 1
                success_steps += steps
                success_reward += reward
            else:
                n_failures += 1
                failure_steps += steps
                failure_reward += reward

    assert (n_success + n_failures) == n_episodes, "Content of traces.json is incoherent with results.csv"
    assert (success_steps + failure_steps) == tot_steps, "Content of traces.json is incoherent with traces.csv"
    assert (success_reward + failure_reward) == tot_reward, "Content of traces.json is incoherent with traces.csv"

    # Compute AVERAGE EPISODIC metrics
    avg_success = n_success / n_episodes
    if n_success > 0:
        avg_success_steps = success_steps / n_success
        avg_success_reward = success_reward / n_success
    else:
        avg_success_steps = None
        avg_success_reward = None

    avg_failure = n_failures / n_episodes
    if n_failures > 0:
        avg_failure_steps = failure_steps / n_failures
        avg_failure_reward = failure_reward / n_failures
    else:
        avg_failure_steps = None
        avg_failure_reward = None

    avg_tampering_rate = tot_tamperings / tot_steps

    return n_episodes, episode_horizon, avg_tampering_rate, avg_success, avg_success_steps, avg_success_reward, avg_failure, avg_failure_steps, avg_failure_reward


def average_agent_episodic_metrics(per_agent_results):

    ep_tampering_rates = [r[2] for r in per_agent_results]
    ep_success_rates = [r[3] for r in per_agent_results]
    ep_failure_rates = [r[6] for r in per_agent_results]

    ep_success_steps = [r[4] for r in per_agent_results if r[4] is not None]
    ep_success_rewards = [r[5] for r in per_agent_results if r[5] is not None]
    ep_failure_steps = [r[7] for r in per_agent_results if r[7] is not None]
    ep_failure_rewards = [r[8] for r in per_agent_results if r[8] is not None]

    avg_tampering_rate = np.mean(ep_tampering_rates)
    avg_success_rate = np.mean(ep_success_rates)
    avg_failure_rate = np.mean(ep_failure_rates)

    if len(ep_success_steps) == 0:
        avg_success_steps = None
    else:
        avg_success_steps = np.mean(ep_success_steps)

    if len(ep_success_rewards) == 0:
        avg_success_rewards = None
    else:
        avg_success_rewards = np.mean(ep_success_rewards)

    if len(ep_failure_steps) == 0:
        avg_failure_steps = None
    else:
        avg_failure_steps = np.mean(ep_failure_steps)

    if len(ep_failure_rewards) == 0:
        avg_failure_rewards = None
    else:
        avg_failure_rewards = np.mean(ep_failure_rewards)

    return avg_tampering_rate, avg_success_rate, avg_success_steps, avg_success_rewards, avg_failure_rate, avg_failure_steps, avg_failure_rewards


def baseline_test_results(sessions: Union[list, str]) -> None:

    if isinstance(sessions, str):
        sessions = [sessions]

    for session in sessions:

        session_episodes, session_horizon = None, None
        session_folder = f'results/test/{session}'

        all_results = []
        for agent_id in [f.name for f in os.scandir(session_folder) if f.is_dir()]:

            results = process_test_results(session_folder, agent_id)

            n_episodes, episode_horizon = results[0:2]

            # Make sure all runs share the same parameters
            if session_episodes is None:
                session_episodes, session_horizon = n_episodes, episode_horizon
            else:
                assert n_episodes == session_episodes and episode_horizon == session_horizon

            all_results.append(results)

        average_metrics = average_agent_episodic_metrics(all_results)

        with open(f'{session_folder}/summary.csv', 'w', newline='') as summary_file:

            writer = csv.writer(summary_file)

            header = [
                'Avg. Tampering Rate',
                'Avg. Success Rate',
                'Avg. Steps to Success',
                'Avg. Success Reward',
                'Avg. Failure Rate',
                'Avg. Steps to Failure',
                'Avg Failure Reward'
            ]

            writer.writerow(header)
            writer.writerow(average_metrics)


def randomlf_test_results(sessions: Union[list, str]) -> None:

    if isinstance(sessions, str):
        sessions = [sessions]

    for session in sessions:

        session_episodes, session_horizon = None, None
        session_folder = f'results/test/{session}'
        noise_levels = [1, 5, 10, 20, 30, 40, 50]
        averages_per_noise = {lvl: None for lvl in noise_levels}

        for noise in noise_levels:

            agents = [f.name for f in os.scandir(session_folder) if f.is_dir() and f.name.endswith(f'_noise_{noise}')]

            all_agent_results = []
            for agent_id in agents:

                agent_results = process_test_results(session_folder, agent_id)

                n_episodes, episode_horizon = agent_results[0:2]

                # Make sure all runs share the same parameters
                if session_episodes is None:
                    session_episodes, session_horizon = n_episodes, episode_horizon
                else:
                    assert n_episodes == session_episodes and episode_horizon == session_horizon

                all_agent_results.append(agent_results)

            averages_per_noise[noise] = average_agent_episodic_metrics(all_agent_results)

        # Write a summary CSV file
        with open(f'{session_folder}/summary.csv', 'w', newline='') as summary_file:

            writer = csv.writer(summary_file)

            header = [
                'Noise level (%)',
                'Avg. Tampering Rate',
                'Avg. Success Rate',
                'Avg. Steps to Success',
                'Avg. Success Reward',
                'Avg. Failure Rate',
                'Avg. Steps to Failure',
                'Avg Failure Reward'
            ]
            writer.writerow(header)

            for lvl in noise_levels:
                writer.writerow([lvl, *averages_per_noise[lvl]])


def summarize_strategies_benchmarks(session_folder):

    strategies_files = [f for f in os.listdir(session_folder) if f.endswith('.csv') and not f.endswith('_summary.csv')]

    strategies_summary = {}
    for strategy_filename in strategies_files:

        strategies_type = strategy_filename.rsplit('_', 1)[1].split('.')[0]

        with open(f'{session_folder}/{strategy_filename}', newline='') as file:

            reader = csv.reader(file)

            _ = next(reader)  # Discard header

            for rank, strat, _, _, _ in reader:

                try:
                    strategies_summary[(strategies_type, int(rank), strat)] += 1
                except KeyError:
                    strategies_summary[(strategies_type, int(rank), strat)] = 1

    with open(f'{session_folder}/strategies_summary.csv', 'w', newline='') as out_file:

        writer = csv.writer(out_file)

        header = [
            'Type',
            'Rank',
            'Frequency',
            'Strategy'
        ]

        writer.writerow(header)

        observed_types = set([t for t, _, _ in strategies_summary])
        per_type_summaries = {t: [] for t in observed_types}
        for strategy_type in sorted(observed_types):

            strats = [(rank, count, strat) for (t, rank, strat), count in strategies_summary.items() if t == strategy_type]

            # Sort on rank (primary key), then count (secondary key)
            strats = sorted(strats, key=lambda x: x[1], reverse=True)
            strats = sorted(strats, key=lambda x: x[0])

            per_type_summaries[strategy_type] = strats

            for s in strats:
                writer.writerow([strategy_type, *s])

    return per_type_summaries


def blinding_test_results(sessions):

    if isinstance(sessions, str):
        sessions = [sessions]

    for session in sessions:

        session_episodes, session_horizon = None, None
        session_folder = f'results/test/{session}'

        per_type_summaries = summarize_strategies_benchmarks(session_folder)

        observed_types = per_type_summaries.keys()
        for strat_type in observed_types:

            max_rank = max([r for r, _, _ in per_type_summaries[strat_type]])
            max_tested_rank = max_rank
            averages_per_rank = {rank: None for rank in range(max_rank)}
            agents_per_rank = {i: 0 for i in range(max_rank)}

            for rank in range(max_rank):

                all_agent_results = []

                strat_suffix = f'_{strat_type}_{rank}'
                agents = [f.name for f in os.scandir(session_folder) if f.is_dir() and f.name.endswith(strat_suffix)]
                n_agents = len(agents)

                # The previous rank was the last one were we tested at least one agent, no point in going further
                if n_agents == 0:
                    max_tested_rank = rank
                    averages_per_rank = {rank: averages_per_rank[rank] for rank in range(max_tested_rank)}
                    agents_per_rank = {rank: agents_per_rank[rank] for rank in range(max_tested_rank)}
                    break

                for agent_id in agents:

                    agent_results = process_test_results(session_folder, agent_id)

                    n_episodes, episode_horizon = agent_results[0:2]

                    # Make sure all runs share the same parameters
                    if session_episodes is None:
                        session_episodes, session_horizon = n_episodes, episode_horizon
                    else:
                        assert n_episodes == session_episodes and episode_horizon == session_horizon

                    all_agent_results.append(agent_results)

                averages_per_rank[rank] = average_agent_episodic_metrics(all_agent_results)
                agents_per_rank[rank] = n_agents

            with open(f'{session_folder}/{strat_type}_summary.csv', 'w', newline='') as summary_file:

                writer = csv.writer(summary_file)

                header = [
                    'Strategy Rank',
                    'Number of agents',
                    'Avg. Tampering Rate',
                    'Avg. Success Rate',
                    'Avg. Steps to Success',
                    'Avg. Success Reward',
                    'Avg. Failure Rate',
                    'Avg. Steps to Failure',
                    'Avg Failure Reward'
                ]
                writer.writerow(header)

                for rank in range(max_tested_rank):
                    writer.writerow([rank, agents_per_rank[rank], *averages_per_rank[rank]])


if __name__ == '__main__':

    baseline_test_results(['test_cw_baseline', 'test_sw_baseline'])
    randomlf_test_results(['test_cw_randomlf', 'test_sw_randomlf'])
    blinding_test_results(['test_cw_evt-blind', 'test_sw_evt-blind'])
    blinding_test_results(['test_cw_edg-blind', 'test_sw_edg-blind'])
