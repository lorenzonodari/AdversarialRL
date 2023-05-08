import random
import time
import os
import csv
import multiprocessing

from environments import CookieWorldEnv
from lrm.agents.run_lrm import run_lrm, original_run_lrm, get_default_lrm_config
from environments.game import Game, GameParams
from environments.grid_world import GridWorldParams
import tensorflow as tf


def save_results(results, session_name, seed):

    results_folder = f'results/{session_name}/seed_{seed}/'

    rewards, rm_scores, rm_info, run_time = results

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

    lp = get_default_lrm_config()

    for i in range(n_runs):

        env = CookieWorldEnv(seed=i)
        env_orig = Game(GameParams(GridWorldParams('cookieworld', 'maps/cookie.txt', 0.05)))

        start = time.time()
        orig_results = original_run_lrm(env_orig, lp, seed=i)
        run_time = int(time.time() - start)
        save_results(orig_results + (run_time,), 'orig_test', seed=i)

        start = time.time()
        results = run_lrm(env, lp, seed=i)
        run_time = int(time.time() - start)
        save_results(results + (run_time,), 'impl_test', seed=i)


if __name__ == '__main__':

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place (Or port to tf2 directly)
    multiprocessing.set_start_method('fork')  # TODO: Find a better place

    check_reimplementation()

