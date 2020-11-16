#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-14-20 10:36
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

"""Miscellaneous models such as ResNet v2 creating and related tools for Keras.
Reference paper:
  - [Identity Mappings in Deep Residual Networks]
    (https://arxiv.org/abs/1603.05027) (CVPR 2016)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras_fn.resnet import lr_schedule, model_depth, resnet_v2


def create_model(model_type, input_shape, num_classes, pretrain=False, depth=None):
    """create_model
    Inputs:
        model_type:
        pretrain:
    Return:
        model: a Keras Model that is not compiled.
    """
    model_types = ["ResNet20v2", "keras.applications.ResNet50V2"]
    if model_type not in model_types:
        raise ValueError('Unknown model_type ' + str(model_type))
    if model_type == model_types[0]:
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
    elif model_type == model_types[1]:
        # weights = "imagenet" if pretrain else None
        weights = None
        if pretrain:
            print("Using ImageNet weights.")
            weights = "imagenet"
        input_ = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
        x = tf.cast(input_, tf.float32)
        x = preprocess_input(x)
        extractor = ResNet50V2(include_top=False, weights=weights,
                               input_shape=input_shape, classes=num_classes)
        x = extractor(x)
        classifier = keras.Sequential([
            keras.Input((4, 4, 2048)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_classes, activation='softmax')],
            name="classifier")
        x = classifier(x)
        model = tf.keras.Model(inputs=[input_], outputs=[x], name="model")

    return model


def main():
    pass


if __name__ == "__main__":
    main()
