import time
import os
import csv
import multiprocessing

from lrm.algorithm import LRMAgent, original_lrm_implementation
from lrm.config import LRMConfig
from environments import CookieWorldEnv
from environments.utils import PerfectRewardMachine, FlattenGridActions
from environments.game import Game, GameParams
from environments.grid_world import GridWorldParams
from labeling import Labeling, MineCountLF, MineSuggestionLF

import gymnasium as gym
import tensorflow as tf
from popgym.envs.minesweeper import MineSweeperMedium
from popgym.wrappers import Antialias, PreviousAction


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

        agent = LRMAgent()
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


def test_minesweeper_lrm(n_runs=5):

    for labeling in [MineSuggestionLF, MineCountLF]:

        for i in range(n_runs):

            agent = LRMAgent(
                rm_u_max=15,
                rm_lr_steps=10,
                rm_tabu_size=int(1e5),

            )

            env = MineSweeperMedium()
            env = PreviousAction(env)
            env = Antialias(env)
            env = Labeling(env, labeling)
            env = FlattenGridActions(env)
            env = PerfectRewardMachine(env, {})

            start = time.time()
            results = agent.run_lrm(env, seed=i)
            run_time = int(time.time() - start)
            save_results(results, run_time, f'test_minesweeper_{labeling.__name__}', seed=i)


if __name__ == '__main__':

    tf.compat.v1.disable_v2_behavior()  # TODO: Find a better place (Or port to tf2 directly)
    multiprocessing.set_start_method('fork')  # TODO: Find a better place

    test_minesweeper_lrm()

    # TODO: Implement proper CLI interface for ease-of-use

