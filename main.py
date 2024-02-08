from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import stable_baselines3 as sb3
import gymnasium as gym
import tensorflow as tf
import wandb
import argparse
import time
import random
import numpy as np

PROJECT_NAME = "RLMaster"


# Fixing the random seed for reproducibility.
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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

    args = parser.parse_args()

    # Initializing the setting.
    run_name = f"{args.env_id}-{args.exp_name}-{args.seed}-{int(time.time())}"
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
