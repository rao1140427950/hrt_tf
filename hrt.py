import tensorflow as tf
from keras.metrics import MeanSquaredError, CosineSimilarity
from keras.layers import BatchNormalization
from utils.losses import ReSpLoss
from utils.dataset import HRTDataset


class HRT:

    def __init__(
            self,
            input_t_shape=(49, 301),
            output_dim=301,
            head_size=128,
            num_heads=4,
            ff_dim=128,
            dropout=0.5,
            num_transformer_blocks=4,
            mlp_units=None,
            regularizer=tf.keras.regularizers.l2(0.0005),
            act='tanh',
            use_bias=True,
            norm_layer=BatchNormalization,
    ):

        if mlp_units is None:
            mlp_units = [ff_dim] * 6
        self._input_t_shape = input_t_shape
        self._head_size = head_size
        self._num_heads = num_heads
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._num_transformer_blocks = num_transformer_blocks
        self._mlp_units = mlp_units
        self._regularizer = regularizer
        self._output_dim = output_dim
        self._act = act
        self._use_bias = use_bias
        self._NormLayer = norm_layer

        self.model = self._build_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.),
            loss={'output_s': ReSpLoss(w_mse=1., w_cos=1., w_std=0.),
                  'output_m': 'mse'},
            loss_weights={'output_s': 1., 'output_m': 1.},
            metrics={'output_s': [CosineSimilarity(), MeanSquaredError()]},
        )


    def _transformer_encoder(self, inputs, masks=None, return_attention_scores=False):
        # inputs: (b, spnum, fdim)
        rst = tf.keras.layers.MultiHeadAttention(
            key_dim=self._head_size, num_heads=self._num_heads, dropout=self._dropout,
            kernel_regularizer=self._regularizer, use_bias=self._use_bias,
        )(inputs, inputs, return_attention_scores=return_attention_scores, attention_mask=masks)  # (b, spnum, fdim)
        if return_attention_scores:
            x, att = rst
        else:
            x = rst
            att = None
        x = tf.keras.layers.Dropout(self._dropout)(x)
        if self._NormLayer is not None:
            x = self._NormLayer(center=self._use_bias)(x)
        res = x + inputs  # (b, 36, fdim)

        x = tf.keras.layers.Conv1D(filters=self._ff_dim, kernel_size=1, padding='same', activation=self._act,
                                   kernel_regularizer=self._regularizer, use_bias=self._use_bias)(res)  # (b, spnum, ff_dim)
        x = tf.keras.layers.Dropout(self._dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, padding='same', activation=self._act,
                                   kernel_regularizer=self._regularizer, use_bias=self._use_bias)(x)
        if self._NormLayer is not None:
            x = self._NormLayer(center=self._use_bias)(x)
        if return_attention_scores:
            return x + res, att
        else:
            return x + res  # (b, 36, fdim)


    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self._input_t_shape, name='input_0')  # (b, spnum, spdim)
        masks = tf.keras.layers.Input(shape=(self._input_t_shape[0], 1), name='input_1')  # (b, spnum, 1)
        raw_t = tf.keras.layers.Input(shape=self._input_t_shape, name='input_2')

        masks_inv = (masks - 1.) * -1.
        att_masks = tf.matmul(masks, masks, transpose_b=True) + tf.matmul(masks_inv, masks_inv, transpose_b=True)

        x = tf.keras.layers.Dense(self._ff_dim, activation=self._act, use_bias=self._use_bias)(inputs)
        if self._NormLayer is not None:
            x = self._NormLayer(center=self._use_bias)(x)
        x = tf.keras.layers.Dense(self._ff_dim, activation=self._act, use_bias=self._use_bias)(x)
        if self._NormLayer is not None:
            x = self._NormLayer(center=self._use_bias)(x)
        x = tf.keras.layers.Dense(self._ff_dim, activation=self._act, use_bias=self._use_bias)(x)
        if self._NormLayer is not None:
            x = self._NormLayer(center=self._use_bias)(x)

        for _ in range(self._num_transformer_blocks):
            x = self._transformer_encoder(x, masks=att_masks)  # (b, spnum, fdim)

        x = tf.reduce_sum(x, axis=1) / tf.reduce_sum(masks, axis=1)

        for dim in self._mlp_units:
            x = tf.keras.layers.Dense(dim, activation=self._act, kernel_regularizer=self._regularizer)(x)
            x = BatchNormalization()(x)
        output_1 = tf.keras.layers.Dense(self._output_dim, kernel_regularizer=self._regularizer, name='output_s')(x)

        x = tf.expand_dims(output_1, axis=-1)
        x = tf.matmul(raw_t, x)
        output_2 = tf.keras.layers.Flatten(name='output_m')(x)

        return tf.keras.Model([inputs, masks, raw_t], [output_1, output_2])


if __name__ == '__main__':
    model = HRT(input_t_shape=(100, 301), norm_layer=None, use_bias=False).model
    val_dataset = HRTDataset(
        spmat_path='srcs/resp_dataset/specs_icvl_d301_val_21k.tfrecords',
        tmat_path='srcs/resp_dataset/tmat_validation_n100d301.tfrecords',
        batch_size=4,
        masks_path='srcs/resp_dataset/masks_n100-16_100k.tfrecords',
        spnum=100,
        min_spnum=16,
        noise_stddev=None,
        sp_amp_range=[0.2, 1.0],
        dict_outputs=True,
    )
    val_samples = val_dataset.generate_dataset()
    for sample in val_samples:
        x, y = sample
        y_pred = model(x)
        input_0 = x['input_0'].numpy()
        input_1 = x['input_1'].numpy()
        input_2 = x['input_2'].numpy()
        output_s = y['output_s'].numpy()
        output_m = y['output_m'].numpy()
        output_s_pred = y_pred[0].numpy()
        output_m_pred = y_pred[1].numpy()
        pass
