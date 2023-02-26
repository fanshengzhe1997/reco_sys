# ================================================================
#   Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Project     : 
#   File name   : config.py
#   Author      : Fan Shengzhe
#   Created date: 2022/5/30 23:45
#   Editor      : PyCharm 2019.1
#   Description :
#
# ================================================================

IMAGE_H = 224
IMAGE_W = 224
BATCH_SIZE = 32 #8
EPOCHS = 300 
LEARNING_RATE = 0.001
DECAY_STEPS = None
DECAY_RATE = 0.96
MODEL_NAME = 'se_resnext_50'#'mobilenet_v3_small'#'efficientnet_v2_xl'# 'se_resnext_101' #'mobilenet_v3'

# DATA_ROOT_PATH = './data'
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
VAL_DIR_NAME = 'val'

IMGS_EXT = 'jpg'

DAILY_CHECKPOINT_DIR = './checkpoints'
BEST_CHECKPOINT_DIR = './best_checkpoints'
CLASS_DICT_DIR = './class_dict'
LOG_DIR = './log'

EXPORT_PATH = './export'

