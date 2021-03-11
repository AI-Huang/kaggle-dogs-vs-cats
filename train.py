#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-08-20 00:29
# @Update  : Mar-04-21 19:29
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import os
import argparse
from datetime import datetime
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, AUC
from utils.dir_utils import makedir_exist_ok
from models.keras_fn.fault_resnet import lr_schedule


def training_args():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--pretrain', type=bool, dest='pretrain',
                        action='store', default=True, help='pretrain, if true, the model will be initialized by pretrained weights.')  # 已经完成的训练数
    parser.add_argument('--start_epoch', type=int, dest='start_epoch',
                        action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=16, help='batch_size, e.g. 16.')  # 16 for Mac, 64, 128 for server
    parser.add_argument('--train_epochs', type=int, dest='train_epochs',
                        action='store', default=150, help='train_epochs, e.g. 150.')  # training 150 epochs to fit enough

    # parser.add_argument('--if_fast_run', type='choice', dest='if_fast_run',
    #   action='store', default=0.99, help='') # TODO

    parser.add_argument('--alpha', type=float, dest='alpha',
                        action='store', default=0.99, help='alpha for focal loss if this loss is used.')

    args = parser.parse_args()
    return args


def main():
    args = training_args()
    if_fast_run = False

    print(f"TensorFlow version: {tf.__version__}.")  # Keras backend

    # paths
    competition_name = "dogs-vs-cats-redux-kernels-edition"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    # Input parameters
    IMAGE_WIDTH = IMAGE_HEIGHT = 128
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    # Load data
    from utils.data_utils import data_generators
    train_generator, validation_generator = data_generators(
        data_dir, target_size=IMAGE_SIZE, batch_size=args.batch_size)

    # Prepare model
    model_type = "ResNet50V2"

    from tensorflow.keras.applications import ResNet50V2
    model = ResNet50V2(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        classes=num_classes
    )

    metrics = [
        Recall(name='recall'),
        Precision(name='precision'),
        TruePositives(name='tp'),  # thresholds=0.5
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        # AUC0(name='auc_good_0'),  # 以 good 为 positive 的 AUC
        AUC(name='auc_bad_1')  # 以 bad 为 positive 的 AUC
    ]

    model.compile(loss=BinaryCrossentropy(),
                  optimizer=Adam(learning_rate=lr_schedule(
                      args.start_epoch)),
                  metrics=metrics)

    # Resume training
    # model_ckpt_file = MODEL_CKPT
    # if os.path.exists(model_ckpt_file):
    #     print("Model ckpt found! Loading...:%s" % model_ckpt_file)
    #     model.load_weights(model_ckpt_file)

    # Prepare model model saving directory.
    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", competition_name)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")  # experiment time
    subfix = os.path.join(model_type, current_time)

    ckpt_dir = os.path.expanduser(os.path.join(prefix, subfix, "ckpts"))
    log_dir = os.path.expanduser(os.path.join(prefix, subfix, "logs"))
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    model_name = "%s.start-%d-epoch-{epoch:03d}-val_loss-{val_loss:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % (
        model_type, args.start_epoch)
    filepath = os.path.join(ckpt_dir, model_name)

    # define callbacks
    # checkpoint = ModelCheckpoint(filepath=filepath, monitor="acc",verbose=1)

    file_writer = tf.summary.create_file_writer(
        log_dir + "/metrics")  # custom scalars
    file_writer.set_as_default()

    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [csv_logger, tensorboard_callback, lr_scheduler]

    # Fit model
    epochs = 3 if if_fast_run else args.train_epochs
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        initial_epoch=args.start_epoch
    )

    # Save last model ckpt
    model.save_weights(f"./{model_type}-last_ckpt.h5")

    # Save history
    with open('./history', 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)


if __name__ == "__main__":
    main()
