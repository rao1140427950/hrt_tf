import numpy as np
import sys
import tensorflow as tf
from scipy.io import loadmat
from keras.metrics import Mean
sys.path.append('..')


class HRTDataset:

    def __init__(
            self,
            spmat_path,
            tmat_path,
            masks_path=None,
            spmat_key='specs',
            tmat_key='Ts',
            spnum=49,
            spdim=301,
            batch_size=128,
            noise_stddev=None,
            sp_amp_range=None,
            min_spnum=None,
            shuffle=True,
            output_measure=False,
            dict_outputs=False,
            t_mean=0.0369,
            t_std=0.0182,
            s_mean=0.2040,
            s_std=0.0840,
            m_mean=0.5999,
            m_std=0.0981,
    ):
        self._spnum = spnum
        self._spdim = spdim

        self._t_dataset = None
        self._s_dataset = None
        self._mask_dataset = None
        self._dict_outputs = dict_outputs

        if type(tmat_path) is list:
            tmat_format = tmat_path[0].split('.')[-1]
        else:
            tmat_format = tmat_path.split('.')[-1]
        if tmat_format == 'mat':
            self._tmat = self._load_dataset_mats(tmat_path, tmat_key)
        elif tmat_format == 'tfrecords':
            self._load_tmat_from_tfrecords(tmat_path)
            self._tmat = None
        else:
            raise ValueError('`tmat` should be .mat or .tfrecords')

        if type(spmat_path) is list:
            tmat_format = spmat_path[0].split('.')[-1]
        else:
            tmat_format = spmat_path.split('.')[-1]
        if tmat_format == 'mat':
            self._spmat = self._load_dataset_mats(spmat_path, spmat_key)
        elif tmat_format == 'tfrecords':
            self._load_smat_from_tfrecords(spmat_path)
            self._spmat = None
        else:
            raise ValueError('`tmat` should be .mat or .tfrecords')

        self._batch_size = batch_size
        self._noise_stddev = noise_stddev
        if sp_amp_range is None:
            self._sp_amp_range = [0.8, 2.0]
        else:
            self._sp_amp_range = sp_amp_range
        self._shuffle = shuffle
        self._random_generator = tf.random.Generator.from_seed(10086)
        self._output_measure = output_measure

        if masks_path is not None:
            self._load_masks_from_tfrecords(masks_path)
            self._masks = None
        else:
            masks = np.ones((100000, spnum, 1), np.float32)
            if min_spnum is not None:
                if min_spnum < spnum:
                    for n in range(100000):
                        idx = np.random.randint(min_spnum, spnum)
                        masks[n, idx:, :] = 0.
                elif min_spnum > spnum:
                    raise ValueError('`min_spnum` should be smaller than `spnum`, but get {} and {}.'.format(min_spnum, spnum))
            self._masks = masks

        self._t_mean = tf.cast(t_mean, dtype=tf.float32)
        self._t_std = tf.cast(t_std, dtype=tf.float32)
        self._s_mean = tf.cast(s_mean, dtype=tf.float32)
        self._s_std = tf.cast(s_std, dtype=tf.float32)
        self._m_mean = tf.cast(m_mean, dtype=tf.float32)
        self._m_std = tf.cast(m_std, dtype=tf.float32)

        self.dataset = None

    @staticmethod
    def _load_dataset_mats(path, key):
        if type(path) is str:
            return loadmat(path)[key]
        elif type(path) is list:
            mats = []
            for p in path:
                mats.append(loadmat(p)[key])
            return np.concatenate(mats, axis=0)
        else:
            raise ValueError('Type of `path` should be str or list, but get {}.'.format(type(path)))

    @staticmethod
    def _norm_max_to_1(data):
        data = data / tf.reduce_max(data)
        return data

    @staticmethod
    def _float_list_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _load_tmat_from_tfrecords(self, tmat_path):
        spectrum_feature_description = {'tmat': tf.io.FixedLenFeature((self._spnum * self._spdim,), tf.float32)}

        def _parse_example_function(example_proto):
            example = tf.io.parse_single_example(example_proto, spectrum_feature_description)
            tmat = example['tmat']
            return tf.reshape(tmat, (self._spnum, self._spdim))

        tmat_dataset = tf.data.TFRecordDataset(tmat_path, buffer_size=1024 * (1024 * 1024))
        self._t_dataset = tmat_dataset.map(_parse_example_function)

    def _load_smat_from_tfrecords(self, smat_path):
        spectrum_feature_description = {'smat': tf.io.FixedLenFeature((self._spdim,), tf.float32)}

        def _parse_example_function(example_proto):
            example = tf.io.parse_single_example(example_proto, spectrum_feature_description)
            smat = example['smat']
            return smat

        smat_dataset = tf.data.TFRecordDataset(smat_path, buffer_size=128 * (1024 * 1024))
        self._s_dataset = smat_dataset.map(_parse_example_function)

    def _load_masks_from_tfrecords(self, masks_path):
        spectrum_feature_description = {'mask': tf.io.FixedLenFeature((self._spnum,), tf.float32)}

        def _parse_example_function(example_proto):
            example = tf.io.parse_single_example(example_proto, spectrum_feature_description)
            mask = example['mask']
            return tf.expand_dims(mask, axis=-1)

        mask_dataset = tf.data.TFRecordDataset(masks_path, buffer_size=128 * (1024 * 1024))
        self._mask_dataset = mask_dataset.map(_parse_example_function)

    def cache_tmat_to_tfrecords(self, save_path):
        n, m, d = np.shape(self._tmat)
        tmats = np.reshape(self._tmat, (n, m * d))
        writer = tf.io.TFRecordWriter(save_path)
        for tmat in tmats:
            feature = {'tmat': self._float_list_feature(tmat)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    def cache_smat_to_tfrecords(self, save_path):
        writer = tf.io.TFRecordWriter(save_path)
        for smat in self._spmat:
            feature = {'smat': self._float_list_feature(smat)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    def cache_masks_to_tfrecords(self, save_path):
        writer = tf.io.TFRecordWriter(save_path)
        for mat in self._masks:
            feature = {'mask': self._float_list_feature(mat.squeeze())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    def generate_dataset(self):
        if self._mask_dataset is None:
            mask_dataset = tf.data.Dataset.from_tensor_slices(self._masks).shuffle(10240,
                                                                                   reshuffle_each_iteration=True)
        else:
            mask_dataset = self._mask_dataset
        if self._s_dataset is None:
            sp_dataset = tf.data.Dataset.from_tensor_slices(self._spmat)
        else:
            sp_dataset = self._s_dataset
        if self._t_dataset is None:
            t_dataset = tf.data.Dataset.from_tensor_slices(self._tmat)
        else:
            t_dataset = self._t_dataset
        if self._shuffle:
            sp_dataset = sp_dataset.shuffle(10240, reshuffle_each_iteration=True)
            t_dataset = t_dataset.shuffle(10240, reshuffle_each_iteration=True)
        sp_t_dataset = tf.data.Dataset.zip((sp_dataset, t_dataset, mask_dataset))
        batch_dataset = sp_t_dataset.batch(self._batch_size)

        mapped_dataset = batch_dataset.map(self._batch_map_func_mask, num_parallel_calls=4)
        self.dataset = mapped_dataset.prefetch(32)
        return self.dataset

    def update_statistics(self, print_=True):
        self._t_mean = tf.cast(0, dtype=tf.float32)
        self._t_std = tf.cast(1, dtype=tf.float32)
        self._s_mean = tf.cast(0, dtype=tf.float32)
        self._s_std = tf.cast(1, dtype=tf.float32)
        self._m_mean = tf.cast(0, dtype=tf.float32)
        self._m_std = tf.cast(1, dtype=tf.float32)
        self._output_measure = True

        t_mean_mean = Mean()
        t_std_mean = Mean()
        s_mean_mean = Mean()
        s_std_mean = Mean()
        m_mean_mean = Mean()
        m_std_mean = Mean()

        if self.dataset is None:
            dataset = self.generate_dataset()
        else:
            dataset = self.dataset
        for sample in dataset:
            (t, mask), sp, measure, _ = sample
            t_mean_mean.update_state(tf.math.reduce_mean(t, axis=(-1, -2)))
            t_std_mean.update_state(tf.math.reduce_std(t, axis=(-1, -2)))
            s_mean_mean.update_state(tf.math.reduce_mean(sp, axis=-1))
            s_std_mean.update_state(tf.math.reduce_std(sp, axis=-1))
            m_mean_mean.update_state(tf.math.reduce_mean(measure, axis=-1))
            m_std_mean.update_state(tf.math.reduce_std(measure, axis=-1))

        self._t_mean = t_mean_mean.result()
        self._t_std = t_std_mean.result()
        self._s_mean = s_mean_mean.result()
        self._s_std = s_std_mean.result()
        self._m_mean = m_mean_mean.result()
        self._m_std = m_std_mean.result()
        self._output_measure = False

        if print_:
            print('t_mean: {:.4f}, t_std: {:.4f}'.format(self._t_mean.numpy(), self._t_std.numpy()))
            print('s_mean: {:.4f}, s_std: {:.4f}'.format(self._s_mean.numpy(), self._s_std.numpy()))
            print('m_mean: {:.4f}, m_std: {:.4f}'.format(self._m_mean.numpy(), self._m_std.numpy()))


    def _batch_map_func_mask(self, sp, t, mask):
        sp = tf.cast(sp, tf.float32)
        spe = tf.expand_dims(sp, axis=-1)
        t = tf.cast(t, tf.float32)
        spe = self._norm_max_to_1(spe)
        spe = spe * self._random_generator.uniform((tf.shape(spe)[0], 1, 1), minval=self._sp_amp_range[0],
                                                   maxval=self._sp_amp_range[1])
        measure = tf.matmul(t, spe)
        t_raw = t
        if self._noise_stddev is not None:
            noise = self._random_generator.normal(tf.shape(measure), stddev=self._noise_stddev)
            measure += noise
        t = tf.math.divide_no_nan(t, measure)

        t = (t - self._t_mean) / self._t_std

        sp = tf.squeeze(spe)
        mask = tf.cast(mask, tf.float32)
        t = t * mask
        measure = measure * mask
        t_raw = t_raw * mask

        if self._dict_outputs:
            return {'input_0': t, 'input_1': mask, 'input_2': t_raw}, {'output_s': sp, 'output_m': tf.squeeze(measure)}

        if self._output_measure:
            return (t, mask), sp, tf.squeeze(measure), t_raw
        else:
            return (t, mask), sp


if __name__ == '__main__':
    train_dataset = HRTDataset(
        spmat_path='../srcs/resp_dataset/specs_icvl_d301_val_21k.mat',
        tmat_path='../srcs/resp_dataset/tmat_validation_10k_d301.mat',
        batch_size=64,
        spnum=49,
        min_spnum=None,
        noise_stddev=None,
        sp_amp_range=[0.2, 1.0],
    )
    # train_dataset.update_statistics()
    train_dataset.cache_smat_to_tfrecords('../srcs/resp_dataset/specs_icvl_d301_val_21k.tfrecords')

    train_dataset = HRTDataset(
        spmat_path='../srcs/resp_dataset/specs_icvl_d301_train_99k.mat',
        tmat_path='../srcs/resp_dataset/tmat_training_100k_d301.tfrecords',
        batch_size=64,
        spnum=49,
        min_spnum=None,
        noise_stddev=None,
        sp_amp_range=[0.2, 1.0],
    )
    # train_dataset.update_statistics()
    train_dataset.cache_smat_to_tfrecords('../srcs/resp_dataset/specs_icvl_d301_train_99k.tfrecords')
    train_dataset.cache_masks_to_tfrecords('../srcs/resp_dataset/masks_n49_full_100k.tfrecords')
