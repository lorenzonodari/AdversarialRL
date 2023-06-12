from matplotlib import pyplot as plt
import numpy as np
import csv
import os

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


def process_training_results(session, execution):
    """
    Process the results obtained from a single training execution.

    :param session: The name of the session for this execution
    :param execution: The id of the given execution e.g: seed_42
    :return rewards, steps, runtime = list of obtained rewards, list of step number for each test, execution time in seconds
    """

    results_folder = os.path.join(session, execution)

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
    plt.suptitle(f'[Session: {session} - Seed: {seed}]')
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


def baseline_test_results(sessions: Union[list, str]) -> None:

    if isinstance(sessions, str):
        sessions = [sessions]

    for session in sessions:

        all_rewards, all_avg_success_steps = [], []
        session_episodes, session_horizon = None, None

        session_folder = f'results/test/{session}'

        for agent_id in [f.name for f in os.scandir(session_folder) if f.is_dir()]:

            with open(f'{session_folder}/{agent_id}/results.csv', newline='') as results_file:

                reader = csv.reader(results_file)

                # Discard header
                _ = next(reader)

                # Get info
                n_episodes, episode_horizon, tot_reward, tot_steps, _, _ = next(reader)

            # Convert to proper data types
            n_episodes = int(n_episodes)
            episode_horizon = int(episode_horizon)
            tot_reward = int(tot_reward)
            tot_steps = int(tot_steps)

            # Make sure all runs share the same parameters
            if session_episodes is None:
                session_episodes, session_horizon = n_episodes, episode_horizon
            else:
                assert n_episodes == session_episodes and episode_horizon == session_horizon

            # Derive additional data
            n_success = tot_reward
            n_failures = n_episodes - n_success
            steps_when_success = tot_steps - (episode_horizon * n_failures)

            # Add results to summary array
            all_rewards.append(tot_reward)
            all_avg_success_steps.append(steps_when_success / n_success)

        avg_reward = np.mean(all_rewards) / session_episodes
        avg_steps = np.mean(all_avg_success_steps)

        with open(f'{session_folder}/summary.csv', 'w', newline='') as summary_file:

            writer = csv.writer(summary_file)
            writer.writerows([
                ['Avg. Episodic Reward', 'Avg. Steps to Success'],
                [avg_reward, avg_steps]
            ])
