from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from deep_q_network import DeepQNetwork
from utils import fix_seed, linear_schedule

import stable_baselines3 as sb3
import gymnasium as gym
import tensorflow as tf
import wandb
import argparse
import time
import random
import numpy as np

PROJECT_NAME = "RLMaster"


# Making an environment.
def make_env(seed, env_id, record, run_name):
    if record:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    env.action_space.seed(seed)
    return env


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--env_id', type=str, required=True, help="The environment to try.")
    parser.add_argument('--exp_name', type=str, default="dqn", help="Any free-from string to indicate the experiment name.")
    parser.add_argument('--track', action='store_true', help="Setting whether to track the training using Wandb or not.")
    parser.add_argument('--record', action='store_true', help="Setting whether to record the video of the agent or not.")
    parser.add_argument('--buffer_size', type=int, default=1e4, help="The size of the replay buffer.")
    parser.add_argument('--total_timesteps', type=int, default=1e7, help="The total timesteps of the experiment.")
    parser.add_argument('--start_eps', type=float, default=1.0, help="The starting value of epslion.")
    parser.add_argument('--end_eps', type=float, default=1e-2, help="The ending value of epsilon.")
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help="The fraction of the steps to adjust epsilon.")

    args = parser.parse_args()

    # Initializing the setting.
    run_name = f"{args.env_id.replace('/', '_')}-{args.exp_name}-{args.seed}-{int(time.time())}"
    if args.track:
        wandb.init(
            project=PROJECT_NAME,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    # TODO: Adding Tensorboard callback.

    fix_seed(args.seed)

    # Setting the environment.
    env = make_env(args.seed, args.env_id, args.record, run_name)
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # Initializing two networks. (Double Deep Q-Learning)
    deep_q_network = DeepQNetwork(num_actions=env.action_space.n)
    target_network = DeepQNetwork(num_actions=env.action_space.n)
    target_network = tf.keras.models.clone_model(deep_q_network)

    # Setting the replay buffer.
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )

    start_time = time.time()
    
    # Main logic.
    obs, _ = env.reset(seed=args.seed)  # obs: (F, W, H)
    for step in range(args.total_timesteps):
        eps = linear_schedule(args.start_eps, args.end_eps, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < eps:  # Exploration.
            action = env.action_space.sample()
        else:  # Exploitation.
            obs = np.expand_dims(np.transpose(np.asarray(obs, dtype=float), (1,2,0)), axis=0)  # (1, W, H, F)
            q_values = deep_q_network(obs)  # (A)
            action = np.argmax(q_values)
        
        # Executing the action and move on.
        next_obs, rewards, terminations, truncations, infos = env.step(action)

        # TODO: Recoding rewards for plotting.

        # TODO: Saving the data into the buffer.

        obs = next_obs
        