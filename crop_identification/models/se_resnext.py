# ================================================================
#   Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Project     : 
#   File name   : se_resnext.py
#   Author      : Fan Shengzhe
#   Created date: 2022/5/31 00:28
#   Editor      : PyCharm 2019.1
#   Description :
#
# ================================================================
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations
def get_group_conv(in_channels,
                   out_channels,
                   kernel_size,
                   strides=(1, 1),
                   padding='valid',
                   groups=1):
    
    if not tf.test.gpu_device_name():
        return GroupConv2D(input_channels=in_channels,
                           output_channels=out_channels,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           groups=groups)
    else:
        return tf.keras.layers.Conv2D(filters=out_channels,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      groups=groups)


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                         kernel_size=kernel_size,
                                                         strides=strides,
                                                         padding=padding,
                                                         data_format=data_format,
                                                         dilation_rate=dilation_rate,
                                                         activation=activations.get(activation),
                                                         use_bias=use_bias,
                                                         kernel_initializer=initializers.get(kernel_initializer),
                                                         bias_initializer=initializers.get(bias_initializer),
                                                         kernel_regularizer=regularizers.get(kernel_regularizer),
                                                         bias_regularizer=regularizers.get(bias_regularizer),
                                                         activity_regularizer=regularizers.get(activity_regularizer),
                                                         kernel_constraint=constraints.get(kernel_constraint),
                                                         bias_constraint=constraints.get(bias_constraint),
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out

    def get_config(self):
        config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "groups": self.groups,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint)
        }
        base_config = super(GroupConv2D, self).get_config()
        return {**base_config, **config}

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = tf.nn.sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = tf.keras.layers.multiply(inputs=[inputs, branch])
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.group_conv = get_group_conv(in_channels=filters,
                                         out_channels=filters,
                                           kernel_size=(3, 3),
                                           strides=strides,
                                           padding="same",
                                           groups=groups)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=2 * filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=2 * filters)

        self.shortcut_conv = tf.keras.layers.Conv2D(filters=2 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.se(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


class SEResNeXt(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality, model_name, num_class):
        super(SEResNeXt, self).__init__()
        self.model_name = model_name

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.block1 = SEResNeXt.__make_layer(filters=128,
                                             strides=1,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[0])
        self.block2 = SEResNeXt.__make_layer(filters=256,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[1])
        self.block3 = SEResNeXt.__make_layer(filters=512,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[2])
        self.block4 = SEResNeXt.__make_layer(filters=1024,
                                             strides=2,
                                             groups=cardinality,
                                             repeat_num=repeat_num_list[3])
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_class,
                                        activation=tf.keras.activations.softmax)

    @staticmethod
    def __make_layer(filters, strides, groups, repeat_num):
        block = tf.keras.Sequential()
        block.add(BottleNeck(filters=filters,
                             strides=strides,
                             groups=groups))
        for _ in range(1, repeat_num):
            block.add(BottleNeck(filters=filters,
                                 strides=1,
                                 groups=groups))

        return block

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)
        return x

    def __repr__(self):
        return "SE_ResNeXt_{}".format(self.model_name)


def get_se_resnext50(num_class):
    return SEResNeXt(repeat_num_list=[3, 4, 6, 3], cardinality=32, model_name="SEResNeXt50", num_class=num_class)


def get_se_resnext101(num_class):
    return SEResNeXt(repeat_num_list=[3, 4, 23, 3], cardinality=32, model_name="SEResNeXt101", num_class=num_class)