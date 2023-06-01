import argparse
import time
import os
import csv
import multiprocessing

from lrm.algorithm import LRMTrainer, original_lrm_implementation
from lrm.config import LRMConfig
from environments import CookieWorldEnv, KeysWorldEnv, SymbolWorldEnv
from environments.game import Game, GameParams
from environments.grid_world import GridWorldParams

import tensorflow as tf


def save_results(results, run_time, session_name, seed):

    results_folder = f'results/{session_name}/seed_{seed}/'

    rewards, rm_scores, rm_info = results

    # Create folder if needed
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save execution time (seconds)
    with open(f'{results_folder}/execution_time.txt', 'w') as time_file:
        time_file.write(str(run_time))

    # Save training rewards
    with open(f'{results_folder}/rewards.csv', 'w', newline='') as rewards_file:
        writer = csv.writer(rewards_file)
        writer.writerow(['Training step', 'Total reward'])
        for step_num, reward in rewards:
            writer.writerow([step_num, reward])

    # Save RM scores
    with open(f'{results_folder}/rm_scores.csv', 'w', newline='') as scores_file:
        writer = csv.writer(scores_file)
        writer.writerow(['Training step', 'Used Traces', 'Total traces', 'Num. examples', 'RM Score', 'Perfect RM Score'])
        for step_num, used_traces, total_traces, n_examples, perfect_score, rm_score in rm_scores:
            writer.writerow([step_num, used_traces, total_traces, n_examples, rm_score, perfect_score])

    # Save final Reward Machine
    with open(f'{results_folder}/reward_machine.txt', 'w') as rm_file:
        for line in rm_info:
            rm_file.write(f'{line}\n')


def check_reimplementation(n_runs=15):

    for i in range(n_runs):

        agent = LRMTrainer()
        config = LRMConfig()

        env = CookieWorldEnv(seed=i)
        env_orig = Game(GameParams(GridWorldParams('cookieworld', 'maps/cookie.txt', 0.05)))

        start = time.time()
        orig_results = original_lrm_implementation(env_orig, config, seed=i)
        run_time = int(time.time() - start)
        save_results(orig_results, run_time, 'orig_test', seed=i)

        start = time.time()
        results = agent.run_lrm(env, seed=i)
        run_time = int(time.time() - start)
        save_results(results, run_time, 'impl_test', seed=i)


def train_cookieworld_lrm_agent(n_runs, session_name, config_file):

    for i in range(n_runs):

        agent = LRMTrainer(f'{session_name}_{i}', config_file=config_file)

        env = CookieWorldEnv(seed=i)

        start = time.time()
        results = agent.run_lrm(env, seed=i)
        run_time = int(time.time() - start)
        save_results(results, run_time, session_name, seed=i)


def train_keysworld_lrm_agent(n_runs, session_name, config_file):

    for i in range(n_runs):

        agent = LRMTrainer(f'{session_name}_{i}', config_file=config_file)

        env = KeysWorldEnv(seed=i)

        start = time.time()
        results = agent.run_lrm(env, seed=i)
        run_time = int(time.time() - start)
        save_results(results, run_time, session_name, seed=i)


def train_symbolworld_lrm_agent(n_runs, session_name, config_file):

    for i in range(n_runs):

        agent = LRMTrainer(f'{session_name}_{i}', config_file=config_file)

        env = SymbolWorldEnv(seed=i)

        start = time.time()
        results = agent.run_lrm(env, seed=i)
        run_time = int(time.time() - start)
        save_results(results, run_time, session_name, seed=i)


def train_lrm_agent(env, n_runs, session_name, config_file):

    scenarios = {

        "CW": train_cookieworld_lrm_agent,
        "KW": train_keysworld_lrm_agent,
        "SW": train_symbolworld_lrm_agent

    }

    if env in scenarios:

        train_function = scenarios[env]
        train_function(n_runs, session_name, config_file)

    else:

        print(f'Requested environment "{env}" not found in available training scenarios')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(description='Train LRM agents of various environments')
    args_parser.add_argument('-e', '--env',
                             choices=['CW', 'KW', 'SW'],  # See train_lrm_agent for the meanings
                             help='The environment to be used for training the LRM agent',
                             required=True)
    args_parser.add_argument('-n', '--n_runs',
                             type=int,
                             help='The number of agents to be trained',
                             required=True)
    args_parser.add_argument('-s', '--session',
                             help='The name of this training session: used to save results',
                             required=True)
    args_parser.add_argument('-c', '--config',
                             help='Path to the configuration file containing LRM and training parameters',
                             type=argparse.FileType('r'),
                             required=True)

    args = args_parser.parse_args()

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place (Or port to tf2 directly)
    multiprocessing.set_start_method('fork')  # TODO: Find a better place

    train_lrm_agent(args.env, args.n_runs, args.session, args.config)

