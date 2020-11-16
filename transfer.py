#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-08-20 00:29
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

"""Transfer learning using pretrain ImageNet model from keras.applications.
Available models: ResNet50, ResNet50V2, etc.
Implemented model: ResNet50V2.
"""
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras_fn.model_utils import create_model
from keras_fn.resnet import lr_schedule, model_depth, resnet_v2
from utils.dir_utils import makedir_exist_ok
from utils.data_utils import data_generators


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument('--gpu', type=int, dest='gpu',
                        action='store', default=0, help='gpu, the number of the gpu used for experiment.')

    # Training parameters
    parser.add_argument('--pretrain', type=int, dest='pretrain',
                        action='store', default=0, help='pretrain, if true, the model will be initialized by pretrained weights.')
    parser.add_argument('--start_epoch', type=int, dest='start_epoch',
                        action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=32, help='batch_size, e.g. 16.')  # 16 for Mac, 32, 64, 128 for server
    parser.add_argument('--epochs', type=int, dest='epochs',
                        action='store', default=150, help='epochs, e.g. 150.')  # training 150 epochs to fit enough

    # parser.add_argument('--if_fast_run', type='choice', dest='if_fast_run',
    #   action='store', default=0.99, help='') # TODO

    parser.add_argument('--alpha', type=float, dest='alpha',
                        action='store', default=0.99, help='alpha for focal loss if this loss is used.')

    args = parser.parse_args()

    # post processing
    if args.pretrain == 0:
        args.pretrain = False
    else:
        args.pretrain = True

    return args


def main():
    args = cmd_parser()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.gpu:], 'GPU')

    if_fast_run = False

    print(f"TensorFlow version: {tf.__version__}.")  # Keras backend
    print(f"Keras version: {keras.__version__}.")
    print("If in eager mode: ", tf.executing_eagerly())
    assert tf.__version__[0] == "2"

    # Prepare model
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    model_type = "two_conv2d_net"
    model_type = 'ResNet%dv%d' % (depth, version)  # "ResNet20v2"

    # or model_type = "keras.applications.ResNet50V2"
    model_type = "keras.applications.ResNet50V2"

    # data path
    competition_name = "dogs-vs-cats-redux-kernels-edition"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    # experiment time
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", competition_name)
    subfix = os.path.join(model_type,
                          '-'.join((date_time, "pretrain", str(args.pretrain))))
    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    log_dir = os.path.expanduser(os.path.join(prefix, "logs", subfix))
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    # Input parameters
    IMAGE_WIDTH = IMAGE_HEIGHT = 128
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    # Data loaders
    train_generator, validation_generator = data_generators(
        data_dir, target_size=image_size, batch_size=args.batch_size)

    # Create model
    model = create_model(model_type, input_shape,
                         num_classes, pretrain=args.pretrain)

    # Compile model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.metrics import Recall, Precision, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, AUC
    metrics = [
        Recall(name='recall'),
        Precision(name='precision'),
        TruePositives(name='tp'),  # thresholds=0.5
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        # AUC0(name='auc_cat_0'),  # 以 cat 为 positive 的 AUC
        AUC(name='auc_dog_1')  # 以 dog 为 positive 的 AUC
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

    # define callbacks
    from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, TensorBoard, ModelCheckpoint
    # model_name = "%s.start-%d-epoch-{epoch:03d}-val_loss-{val_loss:.4f}.h5" % (
    #     model_type, args.start_epoch)
    model_name = "%s-epoch-{epoch:03d}-val_loss-{val_loss:.4f}.h5" % (
        model_type)
    # Prepare model model saving directory.
    filepath = os.path.join(ckpt_dir, model_name)
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor='val_loss', verbose=1)

    file_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "metrics"))  # custom scalars
    file_writer.set_as_default()
    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)
    tensorboard_callback = TensorBoard(
        log_dir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [csv_logger, tensorboard_callback, lr_scheduler, checkpoint]

    # Fit model
    epochs = 3 if if_fast_run else args.epochs
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        initial_epoch=args.start_epoch,
        verbose=1  # 2 for notebook
    )


if __name__ == "__main__":
    main()
