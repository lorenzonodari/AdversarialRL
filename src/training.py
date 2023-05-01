from environments import CookieWorldEnv
from lrm.agents.run_lrm import run_lrm, original_run_lrm, get_default_lrm_config
from environments.game import GameParams
from environments.grid_world import GridWorldParams
import tensorflow as tf

import time
import os
import csv


def save_results(results, session_name, seed):

    results_folder = f'results/{session_name}/seed_{seed}/'

    rewards, rm_scores, rm_info, run_time = results

    # Create folder if needed
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save execution time (seconds)
    with open('execution_time.txt', 'w') as time_file:
        time_file.write(str(run_time))

    # Save training rewards
    with open('rewards.csv', 'w', newline='') as rewards_file:
        writer = csv.writer(rewards_file)
        writer.writerow(['Training step', 'Total reward'])
        for step_num, reward in rewards:
            writer.writerow([step_num, reward])

    # Save RM scores
    with open('rm_scores.csv', 'w', newline='') as scores_file:
        writer = csv.writer(scores_file)
        writer.writerow(['Training step', 'Used Traces', 'Total traces', 'Num. examples', 'RM Score', 'Perfect RM Score'])
        for step_num, used_traces, total_traces, n_examples, perfect_score, rm_score in rm_scores:
            writer.writerow([step_num, used_traces, total_traces, n_examples, rm_score, perfect_score])

    # Save final Reward Machine
    with open('reward_machine.txt', 'w') as rm_file:
        rm_file.write(rm_info)


def check_reimplementation(n_runs=15):

    env = CookieWorldEnv()
    lp = get_default_lrm_config()
    env_orig = GameParams(GridWorldParams('cookie_world', 'maps/cookie.txt', 0.05))

    for i in range(n_runs):

        # TODO: Generate seed and feed it to both implementations
        seed = 0

        start = time.time()
        orig_results = original_run_lrm(env_orig, lp)
        run_time = start - time.time()
        save_results(orig_results + (run_time,), 'orig_test', seed)

        start = time.time()
        results = run_lrm(env)
        run_time = start - time.time()
        save_results(results + (run_time,), 'impl_test', seed)


if __name__ == '__main__':

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place? (Or port to tf2 directly)
    check_reimplementation()


