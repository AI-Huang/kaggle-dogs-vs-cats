#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : Jun-23-19 23:15:04
# @Author  : Kelly Hwong (you@example.org)
# @RefLink : https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
# @RefLink : https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
# @Version : $Id$

import matplotlib.pyplot as plt
import tensorflow as tf
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import tflearn
import os
from utils import create_train_data, create_test_data, create_submission_file, IMG_SIZE
import numpy as np

LR = 1e-4  # 1e-3
EPOCH = 4
BATCH_SIZE = 64

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
MODEL_PATH = os.path.join("./model", MODEL_NAME)

if os.path.exists('train_data.npy'):
    train_data = np.load('train_data.npy', allow_pickle=True)
    print('Train data loaded!')
else:
    train_data = create_train_data()

train = train_data[:-500]
test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

# Learning and finetuning
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('{}.meta'.format(MODEL_PATH)):
    model.load(MODEL_PATH)
    print('model loaded!')

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCH, validation_set=({'input': test_x}, {
          'targets': test_y}), batch_size=BATCH_SIZE, snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_PATH)


overwrite_flag = 0
if os.path.exists('test_data.npy') and overwrite_flag == 0:
    test_data = np.load('test_data.npy', allow_pickle=True)
    print('Test data loaded!')
else:
    test_data = create_test_data()

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

create_submission_file(model, test_data)

data = test_data[1]
img_num = data[1]
img_data = data[0]
data_input4model = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
model_out = model.predict([data_input4model])
print(model_out[0])
print(model_out)
