# ================================================================
#   Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Project     : 
#   File name   : vanilla_cnn.py
#   Author      : Fan Shengzhe
#   Created date: 2022/5/31 00:28
#   Editor      : PyCharm 2019.1
#   Description :
#
# ================================================================
import tensorflow as tf
# Vanilla CNN
def get_vanilla_cnn(num_class):
    return tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, kernel_size=(5, 5)),
                tf.keras.layers.MaxPool2D(strides=2),
                tf.keras.layers.Conv2D(32, kernel_size=(5, 5)),
                tf.keras.layers.MaxPool2D(strides=2),
                tf.keras.layers.Conv2D(64, kernel_size=(5, 5)),
                tf.keras.layers.MaxPool2D(strides=2),
                tf.keras.layers.Conv2D(128, kernel_size=(5, 5)),
                tf.keras.layers.MaxPool2D(strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024),
                tf.keras.layers.Dense(num_class),
                tf.keras.layers.Softmax()
            ])