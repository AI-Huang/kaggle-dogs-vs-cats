#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 07:39
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

# test path
TRAIN_DATA_DIR = "./data/train/"
TEST_DATA_DIR = "./data/test1/"

# constants
IF_FAST_RUN = True
RUN_OVER_NIGHT_EPOCHS = 50
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

BATCH_SIZE = 15


def main():
    """Prepare Data Frame"""
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

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

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

    """Prepare Testing Data"""
    test_filenames = os.listdir(TEST_DATA_DIR)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    # TODO 不知道为什么叫 nb_samples
    nb_samples = test_df.shape[0]

    test_gen = ImageDataGenerator(rescale=1./255)

    """Create Testing Generator"""
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        TEST_DATA_DIR,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )  # Found 12500 images.

    """Predict"""
    model = load_model("./models/model_43-0.9220.h5")

    # predict 要时间
    import time
    start = time.clock()

    predict = model.predict_generator(
        test_generator, steps=np.ceil(nb_samples / BATCH_SIZE))

    elapsed = (time.clock() - start)
    print("Prediction time used:", elapsed)

    np.save("predict.npy", predict)

    """
    # numpy average max
    # test_df['category'] = np.argmax(predict, axis=-1)

    # We will convert the predict category back into our generator classes by using train_generator.class_indices. It is the classes that image generator map while converting data into computer vision
    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)

    # From our prepare data part. We map data with {1: 'dog', 0: 'cat'}. Now we will map the result back to dog is 1 and cat is 0
    test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})
    test_df['category'].value_counts().plot.bar()

    plt.show()
    """

    test_df['category'] = predict[:, 1]
    print("Predict Samples: ")
    print(type(test_df))
    print(test_df.head(10))

    """See predicted result with sample images"""
    """
    sample_test = test_df.head(18)
    sample_test.head()
    plt.figure(figsize=(12, 24))
    for index, row in sample_test.iterrows():
        filename = row['filename']
        category = row['category']
        img = load_img("../input/test1/test1/"+filename, target_size=IMAGE_SIZE)
        plt.subplot(6, 3, index+1)
        plt.imshow(img)
        plt.xlabel(filename + '(' + "{}".format(category) + ')')
    plt.tight_layout()
    plt.show()
    """

    """Submission"""
    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['category']
    submission_df.drop(['filename', 'category'], axis=1, inplace=True)
    submission_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
