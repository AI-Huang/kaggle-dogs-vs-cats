#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-14-20 10:12
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
import argparse
from datetime import datetime
from utils.data_utils import get_test_generator
from keras_fn.model_utils import create_model


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    # Testing parameters
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=32, help='batch_size, e.g. 16.')  # 16 for Mac, 64, 128 for server

    args = parser.parse_args()
    return args


def main():
    args = cmd_parser()

    # data path
    competition_name = "dogs-vs-cats-redux-kernels-edition"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    model_type = "keras.applications.ResNet50V2"

    # experiment time
#     date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    date_time = "20201113-221740"
    ckpt_dir = os.path.expanduser(os.path.join(
        "~", "Documents", "DeepLearningData", competition_name, "ckpts", model_type, date_time))

    # Input parameters
    IMAGE_WIDTH = IMAGE_HEIGHT = 128
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    # Data loader
    test_generator, test_df = get_test_generator(
        data_dir, target_size=image_size, batch_size=args.batch_size)

    model = create_model(model_type, input_shape, num_classes, pretrain=False)

    # Load ckpt
    model.load_weights(os.path.join(
        ckpt_dir, "keras.applications.ResNet50V2-epoch-080-val_loss-0.1708.h5"))

    pred = model.predict(test_generator, workers=4, verbose=1)

    #
    class_indices = {"cat": 0, "dog": 1}
    test_df['label'] = pred[:, class_indices["dog"]]

    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['label']
    submission_df.drop(['filename', 'label'], axis=1, inplace=True)
    submission_path = f"./submissions/submission-{model_type}-{date_time}.csv"
    submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
