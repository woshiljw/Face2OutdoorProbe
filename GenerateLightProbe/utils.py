import tensorflow as tf
import numpy as np
from DaeModel.loadmodel import loadModel


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


def getfeature(datapath, modelpath):
    '''
    extract the feature of thr image of image dataset
    (the method has not tested,you may get error when you load the .npy file)
    :param datapath: image dataset path
    :param modelpath: model weighrs path
    :return: the feature mean of images
    '''
    featureSavePath = 'feature'
    inputdata = tf.placeholder(tf.float32, [None, 32, 128, 3])
    imagedata = np.load(datapath)
    model = loadModel(modelpath)
    features = tf.reduce_mean(model.encoder(inputdata), axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # if your gpu memary is not enough,you can use a cycle language to get the features mean of all images
        features = sess.run(features, feed_dict={inputdata: imagedata})
        np.save(featureSavePath, features)


def shadingmodel(img, faceHdrPath, tmatrixPath):
    I = tf.reshape(img, [-1, 3])
    Sture = tf.constant(np.load(faceHdrPath), dtype=tf.float32)
    tmatrix = tf.constant(np.load(tmatrixPath), dtype=tf.float32)
    r_face = tf.Variable(0.1)
    g_face = tf.Variable(0.1)
    b_face = tf.Variable(0.1)
    r_s = tf.matmul(tmatrix, I[:, 0, tf.newaxis]) * r_face
    g_s = tf.matmul(tmatrix, I[:, 1, tf.newaxis]) * g_face
    b_s = tf.matmul(tmatrix, I[:, 2, tf.newaxis]) * b_face
    return tf.sqrt(tf.square(r_s - Sture[:, 0, tf.newaxis])) + \
           tf.sqrt(tf.square(g_s - Sture[:, 1, tf.newaxis])) + \
           tf.sqrt(tf.square(b_s - Sture[:, 2, tf.newaxis]))
