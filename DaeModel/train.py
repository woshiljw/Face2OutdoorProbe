import tensorflow as tf
import numpy as np

class model():
    def __init__(self,path):
        data = np.stack(np.load(path)[0],axis=0)

    def buildmodel(self,input):
        '''
        this method use to build the DenosingAutoEncoder and initilize the model weights
        :param input: the input pleaseholder
        :return: out value of the model
        '''
        h = tf.layers.conv2d(input, 64, [5, 5], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        h = tf.layers.conv2d(h, 96, [3, 3], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        h = tf.layers.conv2d(h, 96, [3, 3], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        h = tf.layers.conv2d(h, 64, [3, 3], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        h = tf.layers.flatten(h)
        h = tf.layers.dense(h,64)








if __name__ == '__main__':
    model('../data/data.npy')