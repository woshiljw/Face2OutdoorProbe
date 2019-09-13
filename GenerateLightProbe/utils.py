import tensorflow as tf
import numpy as np


def rotateImg(img, ty):
    '''
    This function used to rotate the image
    :param img: input image
    :param ty: the angle of image should be rotated
    :return: rotated image
    '''
    H = 16  # half hight of input img
    W = 32  # half width of input img
    data_type = tf.float32
    x = tf.range(-H, H, dtype=data_type)[:, tf.newaxis] / H
    y = tf.range(-W, W, dtype=data_type)[tf.newaxis, :] / W
    m = tf.reshape(tf.matmul(x, tf.ones_like(y)), [-1, 1])
    n = tf.reshape(tf.matmul(tf.ones_like(x), y), [-1, 1])
    xs = tf.matmul(m, tf.ones_like(tf.transpose(m)))
    ys = tf.matmul(n + ty / np.pi, tf.ones_like(tf.transpose(n)))
    y_s = tf.matmul(n + ty / np.pi - 2, tf.ones_like(tf.transpose(n)))
    flatimg = tf.transpose(tf.reshape(img, [-1, 3]))
    binary = tf.nn.relu(1 - tf.abs(xs - tf.transpose(m) * H) * tf.nn.relu(1 - tf.abs(ys - tf.transpose(n))) * W)
    binary_ = tf.nn.relu(1 - tf.abs(xs - tf.transpose(m) * H) * tf.nn.relu(1 - tf.abs(y_s - tf.transpose(n))) * W)
    r = tf.reduce_sum(flatimg[0] * binary, axis=1) + tf.reduce_sum(flatimg[0] * binary_, axis=1)
    g = tf.reduce_sum(flatimg[1] * binary, axis=1) + tf.reduce_sum(flatimg[1] * binary_, axis=1)
    b = tf.reduce_sum(flatimg[2] * binary, axis=1) + tf.reduce_sum(flatimg[2] * binary_, axis=1)
    outImg = tf.stack([r, g, b], axis=1)

    return outImg
