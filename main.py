from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from tqdm import tqdm
from deep_q_network import DeepQNetwork
from evaluate import evaluate
from utils import fix_seed, linear_schedule, convert_into_tensor, convert_shape
from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer
from tf_agents.specs.tensor_spec import to_nest_array_spec

import gymnasium as gym
import tensorflow as tf
import wandb
import argparse
import time
import random
import numpy as np

PROJECT_NAME = "RLMaster"
NUM_ENVS = 1  # For this project, the number of environment would be 1 for simplicity. However, the implementation is based on vectorized environment.

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Setting the GPU memory.
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5120)])
    logical_gpus = tf.config.list_logical_devices('GPU')
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Making an environment.
def make_env(idx, seed, env_id, record, run_name):
    def func():
        if record and idx == 0:
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

    return func


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Arguments for the setting.
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--env_id', type=str, required=True, help="The environment to try.")
    parser.add_argument('--exp_name', type=str, default="dqn", help="Any free-from string to indicate the experiment name.")
    parser.add_argument('--track', action='store_true', help="Setting whether to track the training using Wandb or not.")
    parser.add_argument('--record', action='store_true', help="Setting whether to record the video of the agent or not.")
    parser.add_argument('--buffer_size', type=int, default=1e4, help="The size of the replay buffer.")
    parser.add_argument('--plotting_frequency', type=int, default=100, help="The frequency of step for plotting into the Tensorboard.")
    parser.add_argument('--save_model', action='store_true', help="Setting whether to save the trained model.")

    # Arguments for trianing the policy.
    parser.add_argument('--total_timesteps', type=int, default=1e7, help="The total timesteps of the experiment.")
    parser.add_argument('--start_eps', type=float, default=1.0, help="The starting value of epslion.")
    parser.add_argument('--end_eps', type=float, default=1e-2, help="The ending value of epsilon.")
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help="The fraction of the steps to adjust epsilon.")
    parser.add_argument('--learning_starting_step', type=int, default=8e4, help="The step when the learning will start.")
    parser.add_argument('--training_frequency', type=int, default=4, help="The frequency of step for training.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size for training.")
    parser.add_argument('--gamma', type=float, default=0.99, help="The discount factor.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="The learning rate for optimization.")
    parser.add_argument('--update_frequency', type=int, default=1000, help="The frequency of step for updating the target network.")

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

    # Adding the Tensorboard writer.
    logdir = f"logs/{run_name}"
    writer = tf.summary.create_file_writer(logdir)

    fix_seed(args.seed)

    # Setting the environment.
    envs = gym.vector.AsyncVectorEnv([
        make_env(i, args.seed + i, args.env_id, args.record, run_name) for i in range(NUM_ENVS)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # Initializing two networks.
    deep_q_network = DeepQNetwork(num_actions=envs.single_action_space.n)
    target_network = DeepQNetwork(num_actions=envs.single_action_space.n)
    target_network = tf.keras.models.clone_model(deep_q_network)

    # Setting the replay buffer.
    data_spec = (
        tf.TensorSpec((4, 84, 84), dtype=tf.float32, name='observations'),
        tf.TensorSpec((4, 84, 84), dtype=tf.float32, name='next_observations'),
        tf.TensorSpec((), dtype=tf.int32, name='actions'),
        tf.TensorSpec((), dtype=tf.float32, name='rewards'),
        tf.TensorSpec((), dtype=tf.bool, name='dones'),
    )
    replay_buffer = PyUniformReplayBuffer(
        capacity=NUM_ENVS * args.buffer_size,
        data_spec=to_nest_array_spec(data_spec)
    )

    loss_func = tf.keras.losses.MeanSquaredError()
    optim = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    start_time = time.time()
    
    # Main logic.
    obs, _ = envs.reset(seed=args.seed)  # obs: (N, F, W, H)
    print("Running the loop...")
    for global_step in tqdm(range(args.total_timesteps)):
        eps = linear_schedule(args.start_eps, args.end_eps, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < eps:  # Exploration.
            actions = np.array([envs.single_action_space.sample() for i in range(envs.num_envs)])
        else:  # Exploitation.
            q_values = deep_q_network.predict_on_batch(convert_into_tensor(obs))  # (N, A)
            actions = tf.math.argmax(q_values, axis=1).numpy()  # (N)
        
        # Executing the action and move on.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Recoding rewards for plotting.
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    for i in range(NUM_ENVS):
                        print(f"global_step={global_step}, episodic_return={info['episode']['r'][i]}")
                        with writer.as_default():
                            tf.summary.scalar(f"env{i}/episodic_return", info["episode"]["r"][i], step=global_step)
                            tf.summary.scalar(f"env{i}/episodic_length", info["episode"]["l"][i], step=global_step)

        # Saving the observation into the buffer.
        next_obs_copied = next_obs.copy()
        for i, truncation in enumerate(truncations):
            if truncation:
                next_obs_copied[i] = infos['final_observation'][i]
        replay_buffer.add_batch((obs, next_obs_copied, actions, rewards, terminations))

        obs = next_obs

        # Training part.
        if global_step >= args.learning_starting_step:
            if global_step % args.training_frequency == 0:
                data = replay_buffer.get_next(sample_batch_size=args.batch_size)

                max_next_vals = tf.reduce_max(target_network.predict_on_batch(tf.convert_to_tensor(convert_shape(data[1]))), axis=1)  # (B)
                td_targets = data[3] + args.gamma * max_next_vals * (1.0 - data[4])  # (B)

                with tf.GradientTape() as tape:
                    original_vals = tf.gather(deep_q_network(tf.convert_to_tensor(convert_shape(data[0]))), data[2], axis=1)  # (B)
                    loss = loss_func(td_targets, original_vals)

                grads = tape.gradient(loss, deep_q_network.trainable_variables)
                optim.apply_gradients(zip(grads, deep_q_network.trainable_variables))

                # Tracking the training.
                if global_step % args.plotting_frequency == 0:
                    mean_q_value = tf.get_static_value(tf.reduce_mean(original_vals))
                    print(f"global_step={global_step}, loss={loss}, Q value={mean_q_value}")
                    with writer.as_default():
                        tf.summary.scalar(f"training/losses", loss, global_step)
                        tf.summary.scalar(f"training/q_values", mean_q_value, global_step)

            # Updating the target network.
            if global_step % args.update_frequency == 0:
                target_network = tf.keras.models.clone_model(deep_q_network)

    # Saving the trained model.
    if args.save_model:
        ckpt_path = f"checkpoints/{run_name}/{args.exp_name}-dqn.ckpt"
        deep_q_network.save_weights(ckpt_path)
        print(f"A new model has been saved to {ckpt_path}.")

        episodic_returns = evaluate(
            deep_q_network,
            ckpt_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            epsilon=0.05,
            record=args.record
        )
        for idx, episodic_return in enumerate(episodic_returns):
            tf.summary.scalar("eval/episodic_return", episodic_return, idx)
