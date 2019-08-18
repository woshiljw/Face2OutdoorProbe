import tensorflow as tf
import numpy as np
from DaeModel.trainConfig import *

class model():
    def __init__(self,path):
        self.data = np.load(path)
        assert self.data.shape == (5540,32,128,3), 'Chack the dataset !'
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
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)

        h = tf.layers.dense(h,1024)
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.reshape(h,[-1,2,8,64])

        h = tf.layers.conv2d(h, 64, [3, 3], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = self.upscale(h)

        h = tf.layers.conv2d(h, 96, [3, 3], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = self.upscale(h)

        h = tf.layers.conv2d(h, 96, [3, 3], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = self.upscale(h)

        h = tf.layers.conv2d(h, 64, [5, 5], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = self.upscale(h)

        h = tf.layers.conv2d(h, 3, [1, 1], padding='SAME')
        h = tf.layers.batch_normalization(h)
        h = tf.nn.sigmoid(h)

        return h

    def upscale(self,input):
        '''
        this method use to  upscale a image
        inputImageSize:[B,H,W,C]    --->     outputImageSize:[B,H*2,W*2,C]
        :param input: the input image
        :return: the upscaled image
        '''
        B,H,W,C = input.get_shape()
        return tf.image.resize_nearest_neighbor(input,[H*2,W*2])

    def train(self,learning_rate,epoch):
        inputdata = tf.placeholder(tf.float32,[None,32,128,3])
        out = self.buildmodel(inputdata)
        error = tf.reduce_mean(tf.abs(inputdata-out))
        opt = tf.train.AdamOptimizer(learning_rate).minimize(error)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epoch):
                index = np.arange(len(self.data))
                data = self.data[np.random.shuffle(index)][0]
                trainingdata = data[:5400]
                testingdata = data[5400:]
                for i in range(54):
                    sess.run(opt,feed_dict={inputdata:trainingdata[i*100:(i+1)*100]})
                print('In epoch {} , loss is {}'.format(e,sess.run(error,feed_dict={inputdata:testingdata})))

if __name__ == '__main__':
    dae = model(datapath)
    dae.train(learning_rate,epoch)
