# ================================================================
#   Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Project     : 
#   File name   : model_builder.py
#   Author      : Fan Shengzhe
#   Created date: 2022/5/31 00:28
#   Editor      : PyCharm 2019.1
#   Description :
#
# ================================================================
import tensorflow as tf
from config import *
from models.vanilla_cnn import get_vanilla_cnn
from models.ghost_efficientnet_v2 import get_ghost_efficientnet_v2_s, get_ghost_efficientnet_v2_m, get_ghost_efficientnet_v2_l, get_ghost_efficientnet_v2_xl
from models.efficientnet_v2 import get_efficientnet_v2_s, get_efficientnet_v2_m, get_efficientnet_v2_l, get_efficientnet_v2_xl
from models.mobilenet_v3 import get_mobilenet_v3_small, get_mobilenet_v3_large
from models.se_resnext import get_se_resnext50, get_se_resnext101

def get_model(model_name, height, width, channels, num_class):
    vanilla_cnn = ['vanilla_cnn']
    ghost_efficientnet_v2 = ['ghost_efficientnet_v2_s', 'ghost_efficientnet_v2_m', 'ghost_efficientnet_v2_l', 'ghost_efficientnet_v2_xl']
    efficientnet_v2 = ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'efficientnet_v2_xl']
    mobilenet_v3 = ['mobilenet_v3_small', 'mobilenet_v3_large']
    se_resnext = ['se_resnext_50', 'se_resnext_101']
    model_list = vanilla_cnn + ghost_efficientnet_v2 + efficientnet_v2 + mobilenet_v3 + se_resnext
    
    if model_name not in model_list:
        raise Exception(f"unsupport model, model_name must be: {model_list}")
        return 
    if model_name == 'vanilla_cnn':
        return get_vanilla_cnn(num_class)
    if model_name == 'ghost_efficientnet_v2_s':
        return get_ghost_efficientnet_v2_s(num_class=num_class)
    if model_name == 'ghost_efficientnet_v2_m':
        return get_ghost_efficientnet_v2_m(num_class=num_class)
    if model_name == 'ghost_efficientnet_v2_l':
        return get_ghost_efficientnet_v2_l(num_class=num_class)
    if model_name == 'ghost_efficientnet_v2_xl':
        return get_ghost_efficientnet_v2_xl(num_class=num_class)
    if model_name == 'efficientnet_v2_s':
        return get_efficientnet_v2_s(num_class=num_class)
    if model_name == 'efficientnet_v2_m':
        return get_efficientnet_v2_m(num_class=num_class)
    if model_name == 'efficientnet_v2_l':
        return get_efficientnet_v2_l(num_class=num_class)
    if model_name == 'efficientnet_v2_xl':
        return get_efficientnet_v2_xl(num_class=num_class)
    if model_name == 'mobilenet_v3_small':
        return get_mobilenet_v3_small(num_class)
    if model_name == 'mobilenet_v3_large':
        return get_mobilenet_v3_large(num_class)
    if model_name == 'se_resnext_50':
        return get_se_resnext50(num_class)   
    if model_name == 'se_resnext_101':
        return get_se_resnext101(num_class)