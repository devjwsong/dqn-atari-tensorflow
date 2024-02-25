from typing import Callable
from utils import convert_into_tensor

import tensorflow as tf
import gymnasium as gym
import random
import numpy as np


def evaluate(
    model: tf.keras.Model,
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    epsilon: float = 0.05,
    record: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(0, 0, env_id, record, run_name)])
    model.load_weights(model_path)

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for i in range(envs.num_envs)])
        else:
            q_values = model.predict_on_batch(convert_into_tensor(obs))  # (N, A)
            actions = tf.math.argmax(q_values, axis=1).numpy()  # (N)

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r'].item()}")
                episodic_returns += [info["episode"]["r"].item()]
        obs = next_obs

    return episodic_returns
