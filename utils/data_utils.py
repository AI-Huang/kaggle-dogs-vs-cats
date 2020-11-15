#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-23 23:15:04
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
# @Version : $Id$

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Data loader


def data_generators(data_dir, target_size, batch_size):
    """data_generators
    Inputs:
        data_dir:
        target_size:
        batch_size:
    Return:
        train_generator:
        validation_generator:
    """
    # Prepare DataFrame
    filenames = os.listdir(os.path.join(data_dir, "train"))
    np.random.shuffle(filenames)
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

    # total_train = train_df.shape[0]
    # total_validate = validate_df.shape[0]

    # Training Generator
    # Using real-time data augmentation
    train_datagen = ImageDataGenerator(
        # rescale=1./255,  # rescale=1./255, 因为keras.applications.resnet50.preprocess_input已经做了预处理
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
        target_size=target_size,
        color_mode="rgb",
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    # print(train_generator.class_indices) # {'cat': 0, 'dog': 1}

    # Validation Generator
    valid_datagen = ImageDataGenerator(validation_split=0.2)
    validation_generator = valid_datagen.flow_from_dataframe(
        validate_df,
        directory=os.path.join(data_dir, "train"),
        x_col='filename',
        y_col='label',
        target_size=target_size,
        color_mode="rgb",
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    return train_generator, validation_generator


def get_test_generator(data_dir, target_size, batch_size):
    """get_test_generator
    Inputs:
        data_dir:
        target_size:
        batch_size:
    Return:
        test_generator:
        test_df:
    """
    # Prepare DataFrame
    test_filenames = os.listdir(os.path.join(data_dir, "test"))
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    # num_samples = test_df.shape[0]

    # Test Generator
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        directory=os.path.join(data_dir, "test"),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False
    )  # Found 12500 images.

    return test_generator, test_df
