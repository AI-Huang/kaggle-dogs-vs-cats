#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 06:51
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import AUC, BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from metrics import AUC0

# constants
IF_FAST_RUN = True
RUN_OVER_NIGHT_EPOCHS = 50
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

BATCH_SIZE = 15

METRICS = [
    TruePositives(name='tp'),  # thresholds=0.5
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'),
    AUC0(name='auc_cat_0'),  # 以 good 为 positive 的 AUC
    AUC(name='auc_dog_1')  # 以 bad 为 positive 的 AUC
]


def create_model(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
    """2conv-basic model
    """
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
        IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2, output_dim

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=METRICS)

    return model


def main():
    print(f"TensorFlow version: {tf.__version__}.")
    print(f"Keras version: {keras.__version__}.")
    print("If in eager mode: ", tf.executing_eagerly())
    assert tf.__version__[0] == "2"

    print("Load Config ...")
    with open('../config/config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    print(f"ROOT_PATH: {ROOT_PATH}")
    ROOT_PATH = os.path.expanduser(ROOT_PATH)
    print(f"ROOT_PATH: {ROOT_PATH}")
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])
    print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")

    MODEL_TYPE = "conv2-basic"  # 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = "models-%s/" % MODEL_TYPE
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)
    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)

    print("Create Model...")
    model = create_model(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    model.summary()

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    model_name = "%s-epoch-{epoch:02d}-val_accuracy-{val_accuracy:.4f}.h5" % MODEL_TYPE
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(SAVES_DIR, model_name), monitor="val_accuracy", verbose=1, period=1)
    callbacks = [earlystop, learning_rate_reduction, checkpoint]

    print("Prepare Data Frame")
    filenames = os.listdir(TRAIN_DATA_DIR)
    random.shuffle(filenames)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

    """Sample Image"""
    # sample = random.choice(filenames)
    # image = load_img("./data/train/"+sample)
    # plt.imshow(image)
    # plt.show()

    """Prepare data"""
    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    classes = ["cat", "dog"]

    """Traning Generator"""
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
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    """Validation Generator"""
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    """Example Generation"""
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )

    """Example Generation Ploting"""
    # plt.figure(figsize=(12, 12))
    # for i in range(0, 15):
    #     plt.subplot(5, 3, i+1)
    #     for X_batch, Y_batch in example_generator:
    #         image = X_batch[0]
    #         plt.imshow(image)
    #         break
    # plt.tight_layout()
    # plt.show()

    print("Fit Model")
    epochs = 3 if IF_FAST_RUN else RUN_OVER_NIGHT_EPOCHS
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//BATCH_SIZE,
        steps_per_epoch=10,  # total_train//BATCH_SIZE,
        callbacks=callbacks
    )

    print("Save Model...")
    model.save_weights("./model.h5")


if __name__ == "__main__":
    main()
