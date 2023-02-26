# ================================================================
#   Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Project     : 
#   File name   : utils.py
#   Author      : Fan Shengzhe
#   Created date: 2022/5/31 00:28
#   Editor      : PyCharm 2019.1
#   Description :
#
# ================================================================
import numpy as np
import tensorflow as tf
from config import *
import os
import random


def make_class_dict(data_root_path, train_dir_name, test_dir_name, val_dir_name):
    train_class_list = [class_name.casefold() for class_name in os.listdir(os.path.join(data_root_path, train_dir_name))]
    test_class_list = [class_name.casefold() for class_name in os.listdir(os.path.join(data_root_path, test_dir_name))]
    val_class_list = [class_name.casefold() for class_name in os.listdir(os.path.join(data_root_path, val_dir_name))]
    all_class = list(set(train_class_list) | set(test_class_list) | set(val_class_list))
    class_dict = {}
    rev_class_dict = {}
    for i, class_name in enumerate(all_class):
        class_dict[class_name] = i
        rev_class_dict[i] = class_name
    return class_dict, rev_class_dict
    
    
def get_img_path_list(data_root_path, sub_dir_name, class_dict, one_hot):
    sub_dir_class_list = [class_name for class_name in os.listdir(os.path.join(data_root_path, sub_dir_name))]
    img_path_list = []
    label_list = []
    
    def get_label(img_path, class_dict, one_hot):
        class_name = img_path.split('/')[-2]
        label = class_dict[class_name.casefold()]
        if one_hot:
            label = tf.one_hot(label, len(class_dict))
        return label
    
    for sub_dir_class_name in sub_dir_class_list:
        for img_name in os.listdir(os.path.join(data_root_path, sub_dir_name, sub_dir_class_name)):
            img_path = os.path.join(data_root_path, sub_dir_name, sub_dir_class_name, img_name)
            img_path_list.append(img_path)
            label_list.append(get_label(img_path, class_dict, one_hot))
    img_and_label = list(zip(img_path_list, label_list))
    random.shuffle(img_and_label)
    img_path_list, label_list = zip(*img_and_label)
    return list(img_path_list), list(label_list)


def read_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    return img


def resize(img, height, width):
    img = tf.image.resize(img, [height, width],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def random_flip(img):
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        img = tf.image.flip_left_right(img)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
    return img


def random_brightness(img):
    delta = tf.random.normal((), 0, 50)
    delta = tf.clip_by_value(delta, -5, 50)
    img = tf.image.adjust_brightness(img, delta)
    img = tf.clip_by_value(img, 0, 255)
    return img


def random_hue(img):
    delta = tf.random.normal((), 0, 0.01)
    delta = tf.clip_by_value(delta, -1, 1)
    img = tf.image.adjust_hue(img, delta)
    img = tf.clip_by_value(img, 0, 255)
    return img


def random_saturation(img):
    saturation_factor = tf.random.normal((), 1, 0.01)
    img = tf.image.adjust_saturation(img, saturation_factor)
    img = tf.clip_by_value(img, 0, 255)
    return img


def random_contrast(img):
    contrast_factor = tf.random.normal((), 1, 0.01)
    img = tf.image.adjust_contrast(img, contrast_factor)
    img = tf.clip_by_value(img, 0, 255)
    return img


def random_scale(img):
    factor = tf.clip_by_value(tf.random.normal((), 0, 10), 1, 1.1)
    height, width, _ = img.shape
    offset_h = int(tf.random.uniform((), 0, height * (factor - 1)))
    offset_w = int(tf.random.uniform((), 0, width * (factor - 1)))

    img = tf.image.resize(img, [int(height * factor), int(width * factor)],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, height, width)
    return img


# normalizing the images to [-1, 1]
def normalize(img):
    img = (img / 127.5) - 1
    return img


def transform_train(img_path, label, height, width):
    img = read_img(img_path)
    img = resize(img, height, width)
    img = random_flip(img)
    img = random_scale(img)
    img = random_brightness(img)
    img = random_hue(img)
    img = random_saturation(img)
    img = random_contrast(img)
    img = normalize(img)
    return img, label


def transform_test(img_path, label, height, width):
    img = read_img(img_path)
    img = resize(img, height, width)
    img = normalize(img)
    return img, label
