#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-23 23:15:04
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
# @Version : $Id$

import os

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
# mixing up or currently ordered data that might lead our network astray in training.
from random import shuffle
# a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion
from tqdm import tqdm

TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
IMG_SIZE = 50


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


def create_submission_file(model, test_data):
    with open('./submission/submission_file.csv', 'w') as f:
        f.write('id,label\n')

    with open('./submission/submission_file.csv', 'a') as f:
        for data in tqdm(test_data):
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
            model_out = model.predict([data])[0]
            f.write('{},{}\n'.format(img_num, model_out[1]))
