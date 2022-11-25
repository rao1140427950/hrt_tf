import tensorflow as tf
import os
from hrt import HRT
from utils.dataset import HRTDataset
from utils.callbacks import AdvancedEarlyStopping


def config(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



def train(model=None):
    spnum = 49
    spdim = 301
    ffdim = 128
    numheads = 4
    numblocks = 4
    batch_size = 256
    noise_std = 0.01

    start_epoch = 0
    epoch = 20480
    model_name = 'hrt-49fixed'

    work_dir = './checkpoints/'
    log_dir = work_dir + model_name
    output_model_file = work_dir + model_name + '.h5'
    weight_file = work_dir + model_name + '_weights.h5'
    checkpoint_path = work_dir + 'checkpoint-' + model_name + '.h5'

    if model is None:
        hrt = HRT(
            input_t_shape=(spnum, spdim),
            output_dim=spdim,
            ff_dim=ffdim,
            num_heads=numheads,
            num_transformer_blocks=numblocks,
        )
        model = hrt.model

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        print('Found weights file: {}, load weights.'.format(weight_file))
    else:
        print('No weights file found. Skip loading weights.')

    train_dataset = HRTDataset(
        spmat_path='srcs/resp_dataset/specs_icvl_d301_train_99k.tfrecords',
        tmat_path='srcs/resp_dataset/tmat_training_100k_d301.tfrecords',
        batch_size=batch_size,
        masks_path='srcs/resp_dataset/masks_n49_full_100k.tfrecords',
        spnum=spnum,
        min_spnum=None,
        noise_stddev=noise_std,
        sp_amp_range=[0.2, 1.0],
        dict_outputs=True,
    )
    val_dataset = HRTDataset(
        spmat_path='srcs/resp_dataset/specs_icvl_d301_val_21k.tfrecords',
        tmat_path='srcs/resp_dataset/tmat_validation_10k_d301.tfrecords',
        batch_size=batch_size,
        masks_path='srcs/resp_dataset/masks_n49_full_100k.tfrecords',
        spnum=spnum,
        min_spnum=None,
        noise_stddev=noise_std,
        sp_amp_range=[0.2, 1.0],
        dict_outputs=True,
    )
    train_samples = train_dataset.generate_dataset()
    val_samples = val_dataset.generate_dataset()

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_images=False, write_graph=False)
    early_stop = AdvancedEarlyStopping(patience=256, filter_order=5, decay_rate=10, min_lr=1e-8, log_dir=log_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch'
    )

    model.fit(
        x=train_samples,
        validation_data=val_samples,
        epochs=epoch,
        callbacks=[tensorboard, early_stop, checkpoint],
        initial_epoch=start_epoch,
        shuffle=False,
        verbose=1
    )

    model.save_weights(weight_file)
    model.save(output_model_file, include_optimizer=False)
    return model


if __name__ == '__main__':
    config('0')
    train()