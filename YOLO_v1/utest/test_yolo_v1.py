import YOLO_v1.yolo_v1 as yolo
from scipy.misc import imread, imsave, imresize
import tensorflow as tf
import keras
from keras import layers
import numpy as np


def test_conv2d_LeakyReLU():
    """
    Test conv2d_LeakyReLU function
    :return:
    """
    rst = yolo.conv2d_LeakyReLU(img, 3, 0.1)
    init = tf.global_variables_initializer()
    with tf.Session() as ss:
        ss.run(init)
        rst = ss.run(rst)
        print(rst)


def test_check_input_shape():
    yolo.check_input_shape(img)


def test_yolo_v1_model(inputs):
    yolo.YOLOV1(inputs)


def test_loss():
    y_true = [{'pos': np.array([10, 20, 40, 50]),
               'label': np.zeros([20])
               }]
    y_pred = np.zeros([7, 7, 30])

    loss = yolo.loss(y_true, y_pred)
    print(loss)

if __name__ == "__main__":
    # img_path = '/home/autel/data/exp_imgs/face_448x448.jpg'
    # img = imread(img_path)
    # img = img.reshape([img.shape[0],img.shape[1], img.shape[2]])
    # img = tf.convert_to_tensor(img.astype(float))

    #input = layers.Input(tensor=img)
    # input = layers.Input(shape=(448, 448, 3))
    # print(input.shape, len(input.shape))
    # test_yolo_v1_model(input)

    test_loss()



