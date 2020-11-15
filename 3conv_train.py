#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 06:51
# @Update  : Nov-08-20 00:29
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification


import os
from datetime import datetime
import json
import argparse
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC, BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras_fn.three_conv2d_net import three_conv2d_net
# from keras_fn.metrics import AUC0
from utils.dir_utils import makedir_exist_ok


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    # Training parameters
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

    print(f"TensorFlow version: {tf.__version__}.")
    print(f"Keras version: {keras.__version__}.")
    print("If in eager mode: ", tf.executing_eagerly())
    assert tf.__version__[0] == "2"

    model_type = "three_conv2d_net"

    # data path
    competition_name = "dogs-vs-cats-redux-kernels-edition"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    # experiment time
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.expanduser(
        f"~/Documents/DeepLearningData/{competition_name}/ckpts/{model_type}/{date_time}")
    log_dir = os.path.expanduser(
        f"~/Documents/DeepLearningData/{competition_name}/logs/{model_type}/{date_time}")
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    # Input parameters
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    # Create model
    metrics = [
        TruePositives(name='tp'),  # thresholds=0.5
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        # AUC0(name='auc_cat_0'),  # 以 good 为 positive 的 AUC
        AUC(name='auc_dog_1')  # 以 bad 为 positive 的 AUC
    ]

    model = three_conv2d_net(input_shape=input_shape, metrics=metrics)
    model.summary()

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    model_name = "%s-epoch-{epoch:02d}-val_accuracy-{val_accuracy:.4f}.h5" % model_type
    filepath = os.path.join(ckpt_dir, model_name)
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="val_accuracy", verbose=1, period=1)
    callbacks = [earlystop, learning_rate_reduction, checkpoint]

    # Prepare DataFrame
    filenames = os.listdir(os.path.join(data_dir, "train"))
    random.shuffle(filenames)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        elif category == 'cat':
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

    # Show sample image
    # sample = np.random.choice(filenames)
    # image = load_img("./data/train/"+sample)
    # plt.imshow(image)
    # plt.show()

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    classes = ["cat", "dog"]

    # Traning Generator
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        os.path.join(data_dir, "train"),
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=args.batch_size
    )

    # Validation Generator
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        os.path.join(data_dir, "train"),
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=args.batch_size
    )

    # Example Generation
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        os.path.join(data_dir, "train"),
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )

    # Example Generation Ploting
    # plt.figure(figsize=(12, 12))
    # for i in range(0, 15):
    #     plt.subplot(5, 3, i+1)
    #     for X_batch, Y_batch in example_generator:
    #         image = X_batch[0]
    #         plt.imshow(image)
    #         break
    # plt.tight_layout()
    # plt.show()

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


if __name__ == "__main__":
    main()
