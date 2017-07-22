import tensorflow as tf
import numpy as np

from basic_network import BasicACNetwork
from block_crnn import block_c_rnn_zero_init_without_size
from layers import pixel_wise_softmax_2
from layers import weight_variable
from layers import bias_variable
from layers import conv2d
from vot2016 import VOT2016_Data_Provider

class Conv_Net(BasicACNetwork):
    def __init__(self, name, nx, ny, channels, n_class, filter_size=3, cost="cross_entropy", cost_kwargs={}):
        self.name = name
        BasicACNetwork.__init__(self, self.name)
        self.nx = nx
        self.ny = ny
        self.channels = channels
        self.n_class = n_class
        self.filter_size = filter_size
        self.variables = [] # for regularizer
        with tf.variable_scope(self.name) as vs:
            self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, None, nx, ny, channels])
            self.keep_prob = tf.placeholder(dtype = tf.float32)
            self.predict, self.offset = self._create_net_test()
            self.labels = tf.placeholder(dtype = tf.float32, shape=[None, None, nx - self.offset, ny - self.offset, n_class])
            self.cost = self._get_cost(self.predict, cost, cost_kwargs)
        self.optimizer = None
        
    def refresh_variables(self):
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        #print('==================================')
        #for a in self.vars:
        #    print(a)
        #print('----------------------------------')
        #for a in tf.global_variables():
        #    print(a)
        #print('==================================')

    def _create_net_test(self):
        #x_image = tf.reshape(x, tf.stack([batch_size, steps, nx, ny, channels]))
        batch_size = tf.shape(self.inputs)[0]
        steps = tf.shape(self.inputs)[1]
        in_node = self.inputs
        
        in_size = 1000
        size = in_size
        with tf.variable_scope('r_net_test') as vs:
            in_node = block_c_rnn_zero_init_without_size(self.nx, self.ny, in_node, self.channels, self.channels)
        
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

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.labels, [-1, self.n_class])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(
                    np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits(
                    flat_logits, flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection / (union))

        else:
            raise ValueError("Unknown cost function: " % cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable)
                                for variable in self.variables])
            loss += (regularizer * regularizers)

        return loss

def test_convnet():
    channels = 3
    n_class = 2
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    iptdata, gtdata = dptest.get_data_one_batch(8)
    iptdata_shape = np.shape(iptdata)
    batch_size, steps, nx, ny, channels = iptdata_shape
    net = Conv_Net('test_convnet', nx, ny, channels, n_class)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            net.inputs: iptdata,
            #net.labels: gtdata,
            net.keep_prob: 1.0
        }
        print(sess.run(net.predict, feed_dict=feed_dict))
        

    #net.predict

if __name__ == '__main__':
    test_convnet()