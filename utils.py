import random
import numpy as np
import tensorflow as tf


# Fixing the random seed for reproducibility.
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Adjusting epsilon as the step increases.
def linear_schedule(start, end, duration, step):
    slope = (end - start) / duration
    return max(slope * step + start, end)

# Converting the shape of the observation for processing.
def convert_shape(obs):  # obs: (B, F, W, H)
    return np.transpose(obs, (0,2,3,1))  # (B, W, H, F)


# Converting the observation into the batched NumPy array.
def convert_into_numpy(obs):  # obs: (B, F, W, H)
    return convert_shape(np.asarray(obs, dtype=float))  # (B, W, H, F)


# Converting the observation into the Tensorflow tensor.
def convert_into_tensor(obs):  # obs: (B, F, W, H)
    obs_np = convert_into_numpy(obs)  # (B, W, H, F)
    return tf.convert_to_tensor(obs_np)  # (B, W, H, F)
