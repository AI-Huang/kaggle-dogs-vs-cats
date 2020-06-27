#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from optparse import OptionParser

import tensorflow as tf  # to check backend
# from tensorflow import keras  # if we want tf2 Keras, not standalone Keras
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import Recall, Precision, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, AUC
from resnet.resnet import model_depth, resnet_v2, lr_schedule
from metrics import AUC0

TOTAL_TRAIN = 30000 * 0.8
TOTAL_VALIDATE = 30000 * 0.2

# constants
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]

METRICS = [
    Recall(name='recall'),
    Precision(name='precision'),
    TruePositives(name='tp'),  # thresholds=0.5
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'),
    AUC0(name='auc_good_0'),  # 以 good 为 positive 的 AUC
    AUC(name='auc_bad_1')  # 以 bad 为 positive 的 AUC
]


def cmd_parser():
    parser = OptionParser()
    # Parameters we care
    parser.add_option('--start_epoch', type='int', dest='start_epoch',
                      action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_option('--batch_size', type='int', dest='batch_size',
                      action='store', default=16, help='batch_size, e.g. 16.')  # 16 for Mac, 64, 128 for server
    parser.add_option('--train_epochs', type='int', dest='train_epochs',
                      action='store', default=150, help='train_epochs, e.g. 150.')  # training 150 epochs to fit enough
    # parser.add_option('--if_fast_run', type='choice', dest='if_fast_run',
    #   action='store', default=0.99, help='') # TODO
    parser.add_option('--alpha', type='float', dest='alpha',
                      action='store', default=0.99, help='alpha for focal loss if this loss is used.')

    args, _ = parser.parse_args(sys.argv[1:])
    return args


def main():
    options = cmd_parser()
    if_fast_run = False
    print(f"TensorFlow version: {tf.__version__}.")  # Keras backend
    print(f"Keras version: {keras.__version__}.")
    print("If in eager mode: ", tf.executing_eagerly())
    assert tf.__version__[0] == "2"

    print("Load Config ...")
    with open('./config/config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    print(f"ROOT_PATH: {ROOT_PATH}")
    ROOT_PATH = os.path.expanduser(ROOT_PATH)
    print(f"ROOT_PATH: {ROOT_PATH}")
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])
    print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")

    print("Prepare Model")
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = "models-%s/" % MODEL_TYPE
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)
    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    MODEL_CKPT = os.path.join(
        SAVES_DIR, "ResNet56v2-epoch-149-auc_good_0-0.9882-auc_bad_1-0.9886.h5")  # CONFIG["MODEL_CKPT"]
    print(f"MODEL_CKPT: {MODEL_CKPT}")

    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss=BinaryCrossentropy(),
                  optimizer=Adam(learning_rate=lr_schedule(
                      options.start_epoch)),
                  metrics=METRICS)
    # model.summary()
    print(MODEL_TYPE)

    print("Resume Training...")
    model_ckpt_file = MODEL_CKPT
    if os.path.exists(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    # Prepare model model saving directory.
    model_name = "%s.start-%d-epoch-{epoch:03d}-auc_good_0-{auc_good_0:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % (
        MODEL_TYPE, options.start_epoch)
    filepath = os.path.join(SAVES_DIR, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc_good_0", verbose=1)
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc_good_0",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [learning_rate_reduction, checkpoint]  # 不要 earlystop

    print("Prepare Data Frame...")
    filenames = os.listdir(TRAIN_DATA_DIR)
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
        directory=TRAIN_DATA_DIR,
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=options.batch_size,
        shuffle=True,
        seed=42
    )

    print("Validation Generator...")
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_dataframe(
        validate_df,
        directory=TRAIN_DATA_DIR,
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=options.batch_size,
        shuffle=True,
        seed=42
    )

    print("Fit Model...")
    epochs = 3 if if_fast_run else options.train_epochs
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=TOTAL_VALIDATE//options.batch_size,
        steps_per_epoch=TOTAL_TRAIN//options.batch_size,
        callbacks=callbacks,
        initial_epoch=options.start_epoch
    )

    print("Save Model...")
    model.save_weights("model-" + MODEL_TYPE + ".h5")

    print("Save History...")
    with open('./history', 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)


if __name__ == "__main__":
    main()
