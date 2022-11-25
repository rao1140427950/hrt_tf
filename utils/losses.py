from keras.losses import Loss, cosine_similarity, mean_squared_error
import tensorflow as tf
from utils.math_ops import standardize_tf


class ReSpLoss(Loss):
    def __init__(self, w_mse=1., w_cos=1., w_std=0.1, name='resploss'):
        super().__init__(name=name)
        self._w_mse = tf.cast(w_mse, tf.float32)
        self._w_cos = tf.cast(w_cos, tf.float32)
        self._w_std = tf.cast(w_std, tf.float32)

    def call(self, y_true, y_pred):
        mse_loss = mean_squared_error(y_true, y_pred)
        cos_loss = cosine_similarity(y_true, y_pred)
        std_loss = mean_squared_error(standardize_tf(y_true, axis=-1), standardize_tf(y_pred, axis=-1))
        return self._w_mse * mse_loss + self._w_cos * cos_loss + self._w_std * std_loss

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        if 'w_mse' in config:
            w_mse = config['w_mse']
        else:
            w_mse = 1.
        if 'w_cos' in config:
            w_cos = config['w_cos']
        else:
            w_cos = 1.
        if 'w_std' in config:
            w_std = config['w_std']
        else:
            w_std = 0.1
        if 'name' in config:
            name = config['name']
        else:
            name = 'resploss'
        return cls(w_mse, w_cos, w_std, name)