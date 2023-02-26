# ================================================================
#   Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Project     : 
#   File name   : debugtools.py
#   Author      : Fan Shengzhe
#   Created date: 2022/6/1 17:15
#   Editor      : PyCharm 2019.1
#   Description :
#
# ================================================================

from config import *
from matplotlib import pyplot as plt
from utils import *
from functools import partial
import os

def test_transform():
    cls_dict, rev_cls_dict = make_class_dict(DATA_ROOT_PATH, TRAIN_DIR_NAME, TEST_DIR_NAME, VAL_DIR_NAME)
    for i in range(1):
        tsfm_train = partial(transform_train, class_dict=cls_dict, one_hot=True, height=IMAGE_H, width=IMAGE_W)
        # inp, label = tsfm_train(os.path.join(DATA_ROOT_PATH, TRAIN_DIR_NAME, 'apple_6', 'r0_0.jpg'))
        tsfm_test = partial(transform_test, class_dict=cls_dict, one_hot=True, height=IMAGE_H, width=IMAGE_W)

        inp, label = tsfm_test(os.path.join(DATA_ROOT_PATH, TRAIN_DIR_NAME, 'apple_6', 'r0_0.jpg'))

        # casting to int for matplotlib to show the image
        plt.figure()
        plt.imshow((inp+1)/2)
        print((inp+1)/2)
        print(rev_cls_dict[tf.argmax(label, axis=0).numpy()])
        plt.show()
