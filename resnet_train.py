#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-08-20 00:29
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import sys
import json
import argparse
from datetime import datetime
import random
import pickle
import numpy as np
import pandas as pd
from optparse import OptionParser
from sklearn.model_selection import train_test_split

import tensorflow as tf  # to check backend
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, AUC
from keras_fn.resnet import model_depth, resnet_v2, lr_schedule
from keras_fn.metrics import AUC0
from utils.dir_utils import makedir_exist_ok


def cmd_parser():
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
    args = cmd_parser()
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
    model_type = 'ResNet%dv%d' % (depth, version)

    # experiment time
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # paths
    competition_name = "dogs-vs-cats-redux-kernels-edition"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    ckpt_dir = os.path.expanduser(
        f"~/Documents/DeepLearningData/{competition_name}/ckpts/{model_type}/{date_time}")
    log_dir = os.path.expanduser(
        f"~/Documents/DeepLearningData/{competition_name}/logs/{model_type}/{date_time}")
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    TOTAL_TRAIN = 30000 * 0.8
    TOTAL_VALIDATE = 30000 * 0.2

    # Input parameters
    IMAGE_WIDTH = IMAGE_HEIGHT = 128
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    print("Prepare Data Frame...")
    filenames = os.listdir(os.path.join(data_dir, "train"))
    random.shuffle(filenames)
    labels = []
    for f in filenames:
        label = f.split('.')[0]
        if label == 'dog':
            labels.append(1)
        elif label == 'cat':
            labels.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'label': labels
    })
    df["label"] = df["label"].replace({0: 'cat', 1: 'dog'})

    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    print("Training Generator...")
    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=os.path.join(data_dir, "train"),
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True,
        seed=42
    )

    print("Validation Generator...")
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_dataframe(
        validate_df,
        directory=os.path.join(data_dir, "train"),
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True,
        seed=42
    )

    # Prepare model
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    model_type = "ResNet%dv%d" % (depth, version)  # "ResNet20v2"
    # or model_type = "keras.applications.ResNet50V2"
    model_type = "keras.applications.ResNet50V2"

    if model_type == "ResNet20v2":
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
    elif model_type == "keras.applications.ResNet50V2":
        weights = "imagenet" if args.pretrain else None
        model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
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
    model_name = "%s.start-%d-epoch-{epoch:03d}-val_loss-{val_loss:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % (
        model_type, args.start_epoch)
    filepath = os.path.join(ckpt_dir, model_name)

    # define callbacks
    # checkpoint = ModelCheckpoint(filepath=filepath, monitor="acc",verbose=1)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        "logs", model_type, current_time)
    makedir_exist_ok(log_dir)

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
