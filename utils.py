import random
import numpy as np
import tensorflow as tf


# Fixing the random seed for reproducibility.
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
