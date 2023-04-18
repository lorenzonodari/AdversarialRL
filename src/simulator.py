import time

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor

from env_utils import PrepareActionsForDQN, DefineRewards


SEED = 42


def parse_cli_args():

    import argparse
    parser = argparse.ArgumentParser(description="Train an RL agent to solve various AI-Collab simulator tasks")
    return parser.parse_args()


def make_simulator_env():

    import gym_collab

    env = gym.make('gym_collab/AICollabWorld-v0',
                   use_occupancy=True,
                   address='https://localhost:50000',
                   view_radius=20,
                   client_number=1)

    env = PrepareActionsForDQN(env)
    env = DefineRewards(env)
    env = FlattenObservation(env)
    env = TimeLimit(env, 25000)
    env = Monitor(env, 'logs/ai_collab_env')

    return env


def train_dqn_agent():

    global SEED

    env = make_simulator_env()

    model = DQN("MlpPolicy", env,
                learning_rate=0.0001,
                buffer_size=25000,
                learning_starts=100000,
                batch_size=256,
                tau=1.0,
                gamma=0.99,
                train_freq=100,
                gradient_steps=1,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                target_update_interval=10000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                tensorboard_log=None,
                policy_kwargs=None,
                verbose=0,
                seed=SEED,
                device='auto',
                _init_setup_model=True)

    logger = configure_logger('logs/dqn.txt', ['stdout', 'log'])
    model.set_logger(logger)

    print(f'\n\n\t[#] Training started')
    print(f'Start time: {time.asctime()}\n\n')

    model.learn(total_timesteps=1000000)

    print(f'\n\n\t[#] Training done')
    print(f'End time: {time.asctime()}\n\n')

    model.save('dqn_ai_collab')

    # obs, info = env.reset()
    # for i in range(100):
    #     action, _states = model.predict(obs)
    #     obs, rewards, terminated, truncated, info = env.step(action)
    #     # env.render()


if __name__ == '__main__':

    args = parse_cli_args()
    train_dqn_agent()
