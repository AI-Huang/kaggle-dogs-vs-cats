#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-06-20 22:15
# @Author  : Kelly Hwong (kan.huang@connect.ust.hk)
# @RefLink : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification


def three_conv2d_net(input_shape, loss="categorical_crossentropy", optimizer="rmsprop", metrics="accuracy"):
    """three_conv2d_net, basic model that consists of 3 Conv2D layers
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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

    model.compile(loss=loss,
                  optimizer=optimizer, metrics=metrics)

    return model


def main():
    pass


if __name__ == "__main__":
    main()
