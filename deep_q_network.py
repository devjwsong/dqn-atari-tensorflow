from email.mime import base
import tensorflow as tf


class DeepQNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.linear2 = tf.keras.layers.Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return self.linear2(x) / 255.0

    def get_config(self):
        base_config = super().get_config()
        config = {'num_actions': self.num_actions}
        return {**base_config, **config}
