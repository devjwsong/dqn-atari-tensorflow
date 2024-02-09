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
