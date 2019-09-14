import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


class loadModel():
    def __init__(self, path):
        '''
        initalize the model parameters
        :param path:the dir path of the saved model
        '''
        ckpt = tf.train.get_checkpoint_state(path)
        ckpt_path = ckpt.model_checkpoint_path
        self.reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)

    def conv(self, input, name):
        kernel = tf.constant(self.reader.get_tensor(name + '/kernel'))
        bias = tf.constant(self.reader.get_tensor(name + '/bias'))
        return tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME') + bias

    def maxpool(self, input):
        return tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    def batchnorm(self, input, name):
        gamma = self.reader.get_tensor(name + '/gamma')
        beta = self.reader.get_tensor(name + '/beta')
        moving_mean = self.reader.get_tensor(name + '/moving_mean')
        moving_variance = self.reader.get_tensor(name + '/moving_variance')
        return tf.layers.batch_normalization(input,
                                             gamma_initializer=tf.constant_initializer(gamma),
                                             beta_initializer=tf.constant_initializer(beta),
                                             moving_mean_initializer=tf.constant_initializer(moving_mean),
                                             moving_variance_initializer=tf.constant_initializer(moving_variance))

    def upsacle(self, input):
        B, H, W, C = input.get_shape()
        return tf.image.resize_nearest_neighbor(input, [H * 2, W * 2])

    def fc(self, input, name):
        kernel = tf.constant(self.reader.get_tensor(name))
        return tf.matmul(input, kernel)

    def encoder(self, input):
        '''
        the parameters of encoder is
        ----------------------------------
        Num.Filters  FilterSize  Resoluton
        ----------------------------------
             64        5x5       32x128
             96        3x3       16x64
             96        3x3       8x32
             64        3x3       4x16
        ----------------------------------
        Full Connect Layer: 64
        the encoder extract the feature(64 dimensions) from a image
        :param input: the imput image
        :return:the feature of the imput image
        '''
        h = self.conv(input, 'conv2d')
        h = self.batchnorm(h, 'batch_normalization')
        h = tf.nn.relu(h)
        h = self.maxpool(h)

        h = self.conv(h, 'conv2d_1')
        h = self.batchnorm(h, 'batch_normalization_1')
        h = tf.nn.relu(h)
        h = self.maxpool(h)

        h = self.conv(h, 'conv2d_2')
        h = self.batchnorm(h, 'batch_normalization_2')
        h = tf.nn.relu(h)
        h = self.maxpool(h)

        h = self.conv(h, 'conv2d_3')
        h = self.batchnorm(h, 'batch_normalization_3')
        h = tf.nn.relu(h)
        h = self.maxpool(h)

        h = self.fc(tf.layers.flatten(h), 'fc1')
        h = self.batchnorm(h, 'batch_normalization_4')
        h = tf.nn.relu(h)
        return h

    def decoder(self, feature):
        '''
        the parameters of encoder is
        Full Connect Layer: 1024
        ----------------------------------
        Num.Filters  FilterSize  Resoluton
        ----------------------------------
             64        3x3       4x16
             96        3x3       8x32
             96        3x3       16x64
             64        5x5       32x128
             3         1x1       32x128
        ----------------------------------
        :param feature: the feature can get from the encoder
        :return: the rebuild image
        '''
        h = self.fc(feature, 'fc2')
        h = self.batchnorm(h, 'batch_normalization_5')
        h = tf.nn.relu(h)
        h = tf.reshape(h, [-1, 2, 8, 64])

        h = self.conv(h, 'conv2d_4')
        h = self.batchnorm(h, 'batch_normalization_6')
        h = tf.nn.relu(h)
        h = self.upsacle(h)

        h = self.conv(h, 'conv2d_5')
        h = self.batchnorm(h, 'batch_normalization_7')
        h = tf.nn.relu(h)
        h = self.upsacle(h)

        h = self.conv(h, 'conv2d_6')
        h = self.batchnorm(h, 'batch_normalization_8')
        h = tf.nn.relu(h)
        h = self.upsacle(h)

        h = self.conv(h, 'conv2d_7')
        h = self.batchnorm(h, 'batch_normalization_9')
        h = tf.nn.relu(h)
        h = self.upsacle(h)

        h = self.conv(h, 'conv2d_8')
        h = self.batchnorm(h, 'batch_normalization_10')
        h = tf.nn.sigmoid(h)

        return h
