# dqn-atari-tensorflow
This is a refactored version of Deep Q-Network[[1]](#1) for Atari which is implemented in TensorFlow.

The original code base is from CleanRL[[2]](#2) which is implemented in PyTorch.

<br/>

---

### Arguments

**Arguments for the basic setting.**

| Argument               | Type         | Description                                                  | Default               |
| ---------------------- | ------------ | ------------------------------------------------------------ | --------------------- |
| `--seed`               | `int`        | The random seed.                                             | `0`                   |
| `--env_id`             | `str`        | The environment to try. (Listed in https://gymnasium.farama.org/environments/atari) | *YOU SHOULD SPECIFY.* |
| `--exp_name`           | `str`        | Any free-form string to indicate the name of this experiment. | *YOU SHOULD SPECIFY.* |
| `--track`              | `store_true` | Setting whether to track the training using Wandb or not.    | -                     |
| `--record`             | `store_true` | Setting whether to record the video of the agent or not.     | *SET BY DEFAULT.*     |
| `--buffer_size`        | `int`        | The size of the replay buffer.                               | `1e4`                 |
| `--plotting_frequency` | `int`        | The frequency of step for plotting into the Tensorboard.     | `100`                 |
| `--save_model`         | `store_true` | Setting whether to save the trained model.                   | *SET BY DEFAULT.*     |

<br/>

| Argument                   | Type    | Description                                                  | Default |
| -------------------------- | ------- | ------------------------------------------------------------ | ------- |
| `--total_timesteps`        | `int`   | The total timesteps of the experiment.                       | `1e7`   |
| `--start_eps`              | `float` | The starting value of epsilon.                               | `1.0`   |
| `--end_eps`                | `float` | The ending value of epsilon.                                 | `1e-2`  |
| `--exploration_fraction`   | `float` | The fraction of the steps to adjust epsilon.                 | `0.1`   |
| `--learning_starting_step` | `int`   | The step when the learning will start.                       | `8e4`   |
| `--training_frequency`     | `int`   | The frequency of step for training.                          | `4`     |
| `--batch_size`             | `int`   | The batch size for training. (This is not the batch size which is put into the buffer!) | `16`    |
| `--gamma`                  | `float` | The discount factor.                                         | `0.99`  |
| `--learning_rate`          | `float` | The learning rate for optimization.                          | `1e-4`  |
| `--update_frequency`       | `int`   | The frequency of step for updating the target network.       | `1000`  |

<br/>

---

### How to run

1. Install the required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Modify the argument by editing `exec_main.sh`.

   <br/>

3. Run the training and see the result!

   ```shell
   sh exec_main.sh
   ```

<br/>

---

<a id="1">[1]</a> Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*. ([https://arxiv.org/pdf/1312.5602.pdf](https://arxiv.org/pdf/1312.5602.pdf))

<a id="2">[2]</a> https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
