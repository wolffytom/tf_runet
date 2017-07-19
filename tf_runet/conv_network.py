import tensorflow as tf
import numpy as np

from basic_network import BasicACNetwork
from block_crnn import block_c_rnn_zero_init_without_size
from layers import *

class Conv_Net(BasicACNetwork):
    def __init__(self, nx, ny, channels, n_class, filter_size=3, keep_prob = 0.5):
        BasicACNetwork.__init__(self, 'Conv_Net')
        self.nx = nx
        self.ny = ny
        self.channels = channels
        self.n_class = n_class
        self.keep_prob = keep_prob
        self.filter_size = filter_size
        with tf.variable_scope('Conv_Net') as name_vs:
            self.input = tf.placeholder(dtype = tf.float32, shape=[None, None, nx, ny, channels])
            self.output, self.offset = self._create_net_test()
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name_vs.name)
        
    def _create_net_test(self):
        #x_image = tf.reshape(x, tf.stack([batch_size, steps, nx, ny, channels]))
        batch_size = tf.shape(self.input)[0]
        steps = tf.shape(self.input)[1]
        in_node = self.input
        
        in_size = 1000
        size = in_size
        with tf.variable_scope('r_net_test') as vs:
            in_node = block_c_rnn_zero_init_without_size(self.nx, self.ny, in_node, self.channels, self.n_class)
    
        with tf.variable_scope('conv1') as vs:
            stddev = np.sqrt(2 / (self.filter_size**2 * self.channels))
            w1 = weight_variable([self.filter_size, self.filter_size, self.channels, self.n_class], stddev)
            b1 = bias_variable([self.n_class])
            in_node = tf.reshape(in_node, shape=[batch_size*steps, self.nx, self.ny, self.channels])
            conv1 = conv2d(in_node, w1, self.keep_prob)
            in_node = tf.nn.relu(conv1 + b1)
            in_node = tf.reshape(in_node, shape=[batch_size, steps, self.nx - 2, self.ny - 2, self.n_class])
            size -= 2
        
        return in_node, int(in_size - size)

def test_convnet():
    nx = 150
    ny = 150
    channels = 3
    n_class = 2
    conv_net = Conv_Net(nx, ny, channels, n_class)
    print(conv_net.vars)

if __name__ == '__main__':
    test_convnet()