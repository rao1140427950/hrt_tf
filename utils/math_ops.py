import numpy as np
import tensorflow as tf


def l2_normalize(data, axis=-1):
    norm = np.linalg.norm(data, ord=2, axis=axis, keepdims=True)
    data = data / norm
    return data

def standardize(data, axis=None):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    data = data - mean
    data = data / (std + 1e-6)
    return data

def standardize_tf(data, axis=None):
    mean = tf.math.reduce_mean(data, axis=axis, keepdims=True)
    std = tf.math.reduce_std(data, axis=axis, keepdims=True)
    data = data - mean
    data = data / (std + 1e-6)
    return data