import tensorflow as tf
import numpy as np
from collections import OrderedDict

from basic_network import BasicACNetwork
from block_crnn import block_c_rnn_zero_init_without_size
from layers import *
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
            self.predict, self.offset = self._create_net()
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

    #@static method
    def _reshape_to_4dim(self, _input):
        shape = tf.shape(_input)
        return tf.reshape(_input, shape=[tf.to_int32(shape[0]*shape[1]), shape[2],shape[3],shape[4]])
    #@static method
    def _reshape_to_5dim(self, _input, batch_size, steps):
        shape = tf.shape(_input)
        return tf.reshape(_input, shape=[batch_size, steps, shape[1],shape[2],shape[3]])

    def _block_rnn(self, in_node, batch_size, steps, sx, sy, channels):
        #print(in_node)
        in_node = self._reshape_to_5dim(in_node, batch_size, steps)
        #print(in_node)
        in_node = block_c_rnn_zero_init_without_size(sx, sy, in_node, channels, channels)
        #print(in_node)
        in_node = self._reshape_to_4dim(in_node)
        #print(in_node)
        return in_node

    def _create_u_net(self, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
        #x_image = tf.reshape(x, tf.stack([batch_size, steps, nx, ny, channels]))
        batch_size = tf.shape(self.inputs)[0]
        steps = tf.shape(self.inputs)[1]
        in_node = self.inputs

        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
    
        in_size = 1000
        size = in_size
        sx = self.nx
        sy = self.ny
        in_node = self._reshape_to_4dim(in_node)
        # down layers
        for layer in range(0, layers):
            features = 2**layer*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, self.channels, features], stddev)
                in_node_channels = self.channels
            else:
                w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
                in_node_channels = features //2
            
            #in_node = self._block_rnn(in_node, batch_size, steps, sx, sy, in_node_channels)
            
            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])
        
            conv1 = conv2d(in_node, w1, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)

            #print('layer:', layer)
            #print(196 * (sx-2) * (sy-2) * features)
            #tmp_h_conv = self._block_rnn(tmp_h_conv, batch_size, steps, sx-2, sy-2, features)

            conv2 = conv2d(tmp_h_conv, w2, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv2 + b2)

            dw_h_convs[layer] = tmp_h_conv#self._block_rnn(tmp_h_conv, batch_size, steps, sx-4, sy-4, features)
        
            #weights.append((w1, w2))
            #biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size -= 4
            sx -= 4
            sy -= 4
            if layer < layers-1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2
                sx = sx//2
                sy = sy//2
        
        in_node = dw_h_convs[layers-1]
        
        # up layers
        for layer in range(layers-2, -1, -1):

            features = 2**(layer+1)*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))

            #in_node = self._block_rnn(in_node, batch_size, steps, sx, sy, features)
        
            wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
            bd = bias_variable([features//2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat
        
            w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
            w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
            b1 = bias_variable([features//2])
            b2 = bias_variable([features//2])
        
            conv1 = conv2d(h_deconv_concat, w1, self.keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            sx = sx*2-2
            sy = sy*2-2

            #print('layer:', layer)
            #print(196 * (sx) * (sy) * (features//2))
            #h_conv = self._block_rnn(h_conv, batch_size, steps, sx, sy, features//2)

            conv2 = conv2d(h_conv, w2, self.keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node
            sx -= 2
            sy -= 2

            #weights.append((w1, w2))
            #biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size *= 2
            size -= 4

        # Output Map
        #in_node = self._block_rnn(in_node, batch_size, steps, sx, sy, features_root)

        weight = weight_variable([1, 1, features_root, self.n_class], stddev)
        bias = bias_variable([self.n_class])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map
        output_map = self._reshape_to_5dim(output_map, batch_size, steps)
        
        return output_map, int(in_size - size)


    def _create_ru_net(self, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
        #x_image = tf.reshape(x, tf.stack([batch_size, steps, nx, ny, channels]))
        batch_size = tf.shape(self.inputs)[0]
        steps = tf.shape(self.inputs)[1]
        in_node = self.inputs

        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
    
        in_size = 1000
        size = in_size
        sx = self.nx
        sy = self.ny
        in_node = self._reshape_to_4dim(in_node)
        # down layers
        for layer in range(0, layers):
            features = 2**layer*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, self.channels, features], stddev)
                in_node_channels = self.channels
            else:
                w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
                in_node_channels = features //2
            
            in_node = self._block_rnn(in_node, batch_size, steps, sx, sy, in_node_channels)
            
            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])
        
            conv1 = conv2d(in_node, w1, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)

            #print('layer:', layer)
            #print(196 * (sx-2) * (sy-2) * features)
            tmp_h_conv = self._block_rnn(tmp_h_conv, batch_size, steps, sx-2, sy-2, features)

            conv2 = conv2d(tmp_h_conv, w2, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv2 + b2)

            dw_h_convs[layer] = self._block_rnn(tmp_h_conv, batch_size, steps, sx-4, sy-4, features)
        
            #weights.append((w1, w2))
            #biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size -= 4
            sx -= 4
            sy -= 4
            if layer < layers-1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2
                sx = sx//2
                sy = sy//2
        
        in_node = dw_h_convs[layers-1]
        
        # up layers
        for layer in range(layers-2, -1, -1):

            features = 2**(layer+1)*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))

            in_node = self._block_rnn(in_node, batch_size, steps, sx, sy, features)
        
            wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
            bd = bias_variable([features//2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat
        
            w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
            w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
            b1 = bias_variable([features//2])
            b2 = bias_variable([features//2])
        
            conv1 = conv2d(h_deconv_concat, w1, self.keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            sx = sx*2-2
            sy = sy*2-2

            #print('layer:', layer)
            #print(196 * (sx) * (sy) * (features//2))
            h_conv = self._block_rnn(h_conv, batch_size, steps, sx, sy, features//2)

            conv2 = conv2d(h_conv, w2, self.keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node
            sx -= 2
            sy -= 2

            #weights.append((w1, w2))
            #biases.append((b1, b2))
            convs.append((conv1, conv2))
        
            size *= 2
            size -= 4

        # Output Map
        in_node = self._block_rnn(in_node, batch_size, steps, sx, sy, features_root)

        weight = weight_variable([1, 1, features_root, self.n_class], stddev)
        bias = bias_variable([self.n_class])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map
        output_map = self._reshape_to_5dim(output_map, batch_size, steps)
        
        return output_map, int(in_size - size)

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
    iptdata = iptdata[:,0:10,:,:,:]
    gtdata = gtdata[:,0:10,:,:,:]
    iptdata_shape = np.shape(iptdata)
    print(iptdata_shape)
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