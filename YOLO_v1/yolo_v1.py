# -*- coding=utf-8 -*-

"""
Create yolo_v1 model.

Date: 2018/03/21
Author: lvhao
"""

import numpy as np
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model


def conv2d_LeakyReLU(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """
    Do 2d-conv with LeakyReLU.
    :param inputs: Input tensor of shape (samples, rows, cols, 3)
    :param filters: Integer, the number of kernels.
    :param kernel: kernel size.
    :param strides: strides.
    :return: Output tensor block.
    """

    x = Conv2D(filters, kernel, strides=strides, padding='same')(inputs)
    x = LeakyReLU(alpha)(x)

    return x


def check_input_shape(inputs):
    """
    Check inputs' type and shape.
    Raise ValueError.
    :return: None
    """

    shape = inputs.shape
    if len(shape) != 4:
        raise ValueError('Input shape is not proper!')

    if shape[1] != 448 or shape[2] != 448 or shape[3] != 3:
        raise ValueError('Input shape should be 448x448x3!')


def YOLOV1(inputs):
    """
    Build YOLO v1 network.
    :param inputs: images of shape (samples, rows, cols, channels)
    :return:
    """
    check_input_shape(inputs)

    alpha = 0.1

    x = conv2d_LeakyReLU(inputs, 64, alpha, (7, 7), strides=2)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    print(x.shape)

    x = conv2d_LeakyReLU(x, 192, alpha, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    print(x.shape)

    x = conv2d_LeakyReLU(x, 128, alpha, (1, 1))
    x = conv2d_LeakyReLU(x, 256, alpha, (3, 3))
    x = conv2d_LeakyReLU(x, 256, alpha, (1, 1))
    x = conv2d_LeakyReLU(x, 512, alpha, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    print(x.shape)

    for _ in range(4):
        x = conv2d_LeakyReLU(x, 256, alpha, (1, 1))
        x = conv2d_LeakyReLU(x, 512, alpha, (3, 3))
    x = conv2d_LeakyReLU(x, 512, alpha, (1, 1))
    x = conv2d_LeakyReLU(x, 1024, alpha, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    print(x.shape)

    for _ in range(2):
        x = conv2d_LeakyReLU(x, 512, alpha, (1, 1))
        x = conv2d_LeakyReLU(x, 1024, alpha, (3, 3))
    x = conv2d_LeakyReLU(x, 1024, alpha, (3, 3))
    x = conv2d_LeakyReLU(x, 1024, alpha, (3, 3), strides=2)
    print(x.shape)

    x = conv2d_LeakyReLU(x, 1024, alpha, (3, 3))
    x = conv2d_LeakyReLU(x, 1024, alpha, (3, 3))
    print(x.shape)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc1')(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.5)(x)
    print(x.shape)

    x = Dense(7 * 7 * 30, name='fc2')(x)
    print(x.shape)

    model = Model(inputs, x, name='YOLO_v1')

    return model

def loss(y_true, y_pred, lambda_c=5, lambda_n=0.5):
    """
    Calculate loss
    :param y_true: ground truth
    :param y_pred: predicted output
    :param lambda_c: loss weight of position item
    :param lambda_n: loss weight of no obj item
    :return: final loss
    """

    grid_side = 7
    grid_len = 448//grid_side
    bbox_num = 2

    # create I to define which grid has object
    I_i = np.zeros([grid_side, grid_side])
    for y_true_data in y_true:
        posi = y_true_data['pos']
        _x, _y, _w, _h = posi[0],posi[1],posi[2],posi[3]
        I_i[_y//grid_len:(_y + _h)//grid_len + 1,
            _x//grid_len:(_x + _w)//grid_len + 1] = 1

    #

    return







