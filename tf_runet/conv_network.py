import tensorflow as tf
import numpy as np
from collections import OrderedDict

from basic_network import BasicACNetwork
from block_crnn import block_c_rnn_zero_init
from block_crnn import block_c_rnn
from block_crnn import block_c_lstmnn
from layers import *
from vot2016 import VOT2016_Data_Provider

class Conv_Net(BasicACNetwork):
    def __init__(self, name, nx, ny, channels, n_class, filter_size=3, cost="cross_entropy", cost_kwargs={}, use_mark = False):
        cost_kwargs["regularizer"] = 0.003
        self.name = name
        #BasicACNetwork.__init__(self, self.name)
        self.nx = nx
        self.ny = ny
        self._calculate_offset()
        self.channels = channels
        self.n_class = n_class
        self.filter_size = filter_size
        self.variables = [] # for regularizer
        self.use_mark = use_mark
        with tf.variable_scope('RU_Net', reuse = tf.AUTO_REUSE):
            self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, None, nx, ny, channels])
            self.labels = tf.placeholder(dtype = tf.float32, shape=[None, None, nx, ny, n_class])
            self.othermarks = tf.placeholder(dtype = tf.float32, shape=[None, None, self.sx, self.sy])
            self.keep_prob = tf.placeholder(dtype = tf.float32)
            self.firstframe = self.inputs[:,:1,:,:,:]
            self.otherframes = self.inputs[:,1:,:,:,:]
            self.firstlabel = self.labels[:,:1,:,:,:]
            self.otherlabels = self.labels[:,1:,self.offsetx:self.offsetx + self.sx,self.offsety:self.offsety + self.sy,:]
            self.predict, self.variables = self._create_ru_net()
            self.cost = self._get_cost(self.predict, self.otherlabels, self.othermarks, "cross_entropy", cost_kwargs)
            self.total_accuracy, self.class_accuracy = self._get_accuracy(self.predict, self.otherlabels)

    #@test
    def _create_net_test(self):
        #x_image = tf.reshape(x, tf.stack([batch_size, steps, nx, ny, channels]))
        batch_size = tf.shape(self.inputs)[0]
        steps = tf.shape(self.inputs)[1]
        in_node = self.inputs
        
        in_size = 1000
        size = in_size
        with tf.variable_scope('r_net_test', reuse = tf.AUTO_REUSE) as vs:
            in_node = block_c_rnn_zero_init(self.nx, self.ny, in_node, self.channels, self.channels)
        
        with tf.variable_scope('conv1', reuse = tf.AUTO_REUSE) as vs:
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
        with tf.variable_scope('reshape_to_4dim', reuse = tf.AUTO_REUSE):
            shape = tf.shape(_input)
            return tf.reshape(_input, shape=[tf.to_int32(shape[0]*shape[1]), shape[2],shape[3],shape[4]])
    #@static method
    def _reshape_to_5dim(self, _input, batch_size, steps):
        with tf.variable_scope('reshape_to_5dim', reuse = tf.AUTO_REUSE):
            shape = tf.shape(_input)
            return tf.reshape(_input, shape=[batch_size, steps, shape[1],shape[2],shape[3]])

    def _block_rnn(self, in_node, batch_size, steps, sx, sy, in_channels, out_channels, initstate, LSTM):
        in_node = tf.reshape(in_node, [batch_size, steps, sx, sy, in_channels])
        if LSTM is True:
            in_node, variables = block_c_lstmnn(sx, sy, in_node, in_channels, initstate, out_channels)
        else:
            in_node, variables = block_c_rnn(sx, sy, in_node, in_channels, initstate, out_channels)
        in_node = tf.reshape(in_node, [batch_size * steps, sx, sy, out_channels])
        return in_node, variables

    def _block_rnn_zero_init(self, in_node, batch_size, steps, sx, sy, channels):
        in_node = self._reshape_to_5dim(in_node, batch_size, steps)
        in_node = block_c_rnn_zero_init(sx, sy, in_node, channels, channels)
        in_node = self._reshape_to_4dim(in_node)
        return in_node

    # test
    def _create_fc_net(self, layers=1, features_root=16, filter_size=3, pool_size=2, summaries=True):
        def _fc_variable(weight_shape):
            with tf.variable_scope('FC_var', reuse = tf.AUTO_REUSE) as vs:
                d = 1.0 / np.sqrt(weight_shape[0])
                bias_shape = [weight_shape[1]]
                bias = tf.get_variable(name = 'b', shape = bias_shape)
                weight = tf.get_variable(name = 'w', shape = weight_shape)
            return weight, bias

        #x_image = tf.reshape(x, tf.stack([batch_size, steps, nx, ny, channels]))
        batch_size = tf.shape(self.inputs)[0]
        steps = tf.shape(self.inputs)[1]
        in_node = tf.reshape(self.inputs, shape=[-1, self.channels])
        w1,b1 = _fc_variable([self.channels, self.n_class])
        output_map = tf.nn.relu(tf.matmul(in_node, w1) + b1)
        output_map = tf.reshape(output_map, [batch_size, steps, self.nx, self.ny, self.n_class])
        
        return output_map, 0

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

    def _calculate_offset(self, layers=3):
        sx = self.nx
        sy = self.ny
        # down layers
        for layer in range(0, layers):
            sx -= 4
            sy -= 4
            if layer < layers-1:
                sx = sx//2
                sy = sy//2
        # up layers
        for layer in range(layers-2, -1, -1):
            sx = sx * 2 - 4
            sy = sy * 2 - 4
        self.sx = sx
        self.offsetx = (self.nx - self.sx) // 2
        self.sy = sy
        self.offsety = (self.ny - self.sy) // 2

    def _create_ru_net(self, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True, LSTM = True):
        first = tf.concat([self.firstframe,self.firstlabel], axis = 4)
        # first.shape is [batch_size, 1, nx, ny, self.channels + self.n_class]
        initstates = {}
        variables = []

        def _ru_part(INIT, in_node):
            batch_size = tf.shape(in_node)[0]
            steps = tf.shape(in_node)[1]

            convs = []
            pools = OrderedDict()
            deconv = OrderedDict()
            dw_h_convs = OrderedDict()
            up_h_convs = OrderedDict()
 
            sx = self.nx
            sy = self.ny
            in_node_channels = -1
            in_node = self._reshape_to_4dim(in_node)

            def block_rnn_part_different_out_channels(name_crnn, in_part, in_part_channels, out_part_channels):
                if INIT is True:
                    #init_part.shape = [batch_size, 1, sx, sy, in_part_channels]
                    initstates[name_crnn] = tf.reshape(in_part, [batch_size, sx, sy, in_part_channels])
                    return in_part
                else:
                    with tf.variable_scope('crnn-'+name_crnn, reuse = tf.AUTO_REUSE) as vs:
                        initstate = initstates[name_crnn]
                        # initstate.shape is [batch_size, sx, sy, state_channels]
                        initstate_shape = initstate.get_shape().as_list()
                        if LSTM is True:
                            state_channels = out_part_channels * 2
                        else:
                            state_channels = out_part_channels
                        assert len(initstate_shape) == 4 and state_channels == initstate_shape[3]
                        out_part, block_rnn_vars = self._block_rnn(in_part, batch_size, steps, sx, sy, in_part_channels, out_part_channels, initstates[name_crnn], LSTM)
                        variables.extend(block_rnn_vars)
                        return out_part
            
            def block_rnn_part(name_crnn, in_part, in_part_channels):
                return block_rnn_part_different_out_channels(name_crnn, in_part, in_part_channels, in_part_channels)
            
            # down layers
            for layer in range(0, layers):
                with tf.variable_scope('down_layer-'+str(layer), reuse = tf.AUTO_REUSE) as vs:
                    features = 2**layer*features_root
                    if LSTM is True and INIT is True:
                        features = features * 2
                    stddev = np.sqrt(2 / (filter_size**2 * features))

                    # block_rnn for input
                    if layer == 0:
                        # in_node.shape is [batch_size, steps, sx, sy, ?]
                        with tf.variable_scope('input_layer'):
                            in_node_ori_channels = (self.channels + self.n_class) if INIT is True else self.channels
                            if LSTM is True:
                                in_node_channels = 2 * (self.channels + self.n_class) if INIT is True else (self.channels + self.n_class)
                            else:
                                in_node_channels = self.channels + self.n_class
                            w_lstminit = weight_variable('w_initfc', [in_node_ori_channels, in_node_channels], stddev)
                            b_lstminit = bias_variable('b_initfc', [in_node_channels])
                            variables.extend((w_lstminit, b_lstminit))
                            in_node = tf.reshape(in_node, [-1, in_node_ori_channels])
                            in_node = tf.nn.relu(tf.matmul(in_node, w_lstminit) + b_lstminit)
                            in_node = tf.reshape(in_node, [batch_size*steps, sx, sy, in_node_channels])
                    else:
                        in_node_channels = features //2
                    in_node = block_rnn_part('down'+str(layer)+'_in', in_node, in_node_channels)
                    #in_node = self._reshape_to_4dim(in_node)
            
                    # conv vars
                    w1 = weight_variable('w1', [filter_size, filter_size, in_node_channels, features], stddev)
                    w2 = weight_variable('w2', [filter_size, filter_size, features, features], stddev)
                    b1 = bias_variable('b1', [features])
                    b2 = bias_variable('b2', [features])
                    variables.extend((w1,b1,w2,b2))
        
                    # conv1 + block_rnn
                    conv1 = conv2d(in_node, w1, self.keep_prob)
                    sx -= 2
                    sy -= 2
                    conv1_relu = tf.nn.relu(conv1 + b1)
                    conv1_relu_r = block_rnn_part('down'+str(layer)+'_conv1_relu', conv1_relu, features)

                    # conv2 + block_rnn
                    conv2 = conv2d(conv1_relu_r, w2, self.keep_prob)
                    sx -= 2
                    sy -= 2
                    conv2_relu = tf.nn.relu(conv2 + b2)
                    conv2_relu_r = block_rnn_part('down'+str(layer)+'_conv2_relu', conv2_relu, features)

                    # for up layers's input.
                    # it should be noticed that this part is the result after an block_rnn opt.
                    dw_h_convs[layer] = conv2_relu_r
                    
                    if layer < layers-1:
                        # maxpool
                        pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                        sx = sx//2
                        sy = sy//2
                        in_node = pools[layer]
        
            in_node = dw_h_convs[layers-1]
        
            # up layers
            for layer in range(layers-2, -1, -1):
                with tf.variable_scope('up-'+str(layer), reuse = tf.AUTO_REUSE) as vs:
                    features = 2**(layer+1)*features_root
                    if LSTM is True and INIT is True:
                        features = features * 2
                    stddev = np.sqrt(2 / (filter_size**2 * features))

                    # conv vars
                    wd = weight_variable('wd', [pool_size, pool_size, features//2, features], stddev)
                    bd = bias_variable('bd', [features//2])
                    w1 = weight_variable('w1', [filter_size, filter_size, features, features//2], stddev)
                    w2 = weight_variable('w2', [filter_size, filter_size, features//2, features//2], stddev)
                    b1 = bias_variable('b1', [features//2])
                    b2 = bias_variable('b2', [features//2])
                    variables.extend((wd,bd,w1,w2,b1,b2))

                    # block_rnn for input
                    in_node = block_rnn_part('up'+str(layer)+'_in', in_node, features)

                    # deconv layer
                    h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                    sx *= 2
                    sy *= 2
                    h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                    deconv[layer] = h_deconv_concat
        
                    # conv1 + block_rnn
                    conv1 = conv2d(h_deconv_concat, w1, self.keep_prob)
                    sx -= 2
                    sy -= 2
                    conv1_relu = tf.nn.relu(conv1 + b1)
                    conv1_relu_r = block_rnn_part('up'+str(layer)+'_conv1_relu', conv1_relu, features//2)

                    # conv2
                    conv2 = conv2d(conv1_relu, w2, self.keep_prob)
                    sx -= 2
                    sy -= 2
                    conv2_relu = tf.nn.relu(conv2+b2)
            
                    up_h_convs[layer] = in_node = conv2_relu

            # last block_rnn
            in_node = block_rnn_part('last_block_rnn', in_node, features//2)

            # returns of this part func
            if True == INIT:
                return None
            else:
                weight = weight_variable('weight', [1, 1, features_root, self.n_class], stddev)
                bias = bias_variable('bias', [self.n_class])
                variables.extend((weight, bias))
                conv = conv2d(in_node, weight, tf.constant(1.0))
                output_map = tf.nn.relu(conv + bias)
                up_h_convs["out"] = output_map
                output_map = self._reshape_to_5dim(output_map, batch_size, steps)

                # softmax
                output_map = tf.nn.softmax(output_map)
                return output_map

        # calculate the initstates
        with tf.variable_scope('init_frame', reuse = tf.AUTO_REUSE) as vs:
            _ru_part(True, first)

        # process other frames
        with tf.variable_scope('other_frames', reuse = tf.AUTO_REUSE) as vs:
            return _ru_part(False, self.otherframes), variables

    def _create_ru_net_zero_init(self, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
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
            
            in_node = self._block_rnn_zero_init(in_node, batch_size, steps, sx, sy, in_node_channels)
            
            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])
        
            conv1 = conv2d(in_node, w1, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)

            tmp_h_conv = self._block_rnn_zero_init(tmp_h_conv, batch_size, steps, sx-2, sy-2, features)

            conv2 = conv2d(tmp_h_conv, w2, self.keep_prob)
            tmp_h_conv = tf.nn.relu(conv2 + b2)

            dw_h_convs[layer] = self._block_rnn_zero_init(tmp_h_conv, batch_size, steps, sx-4, sy-4, features)
        
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

            in_node = self._block_rnn_zero_init(in_node, batch_size, steps, sx, sy, features)
        
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

            h_conv = self._block_rnn_zero_init(h_conv, batch_size, steps, sx, sy, features//2)

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
        in_node = self._block_rnn_zero_init(in_node, batch_size, steps, sx, sy, features_root)

        weight = weight_variable([1, 1, features_root, self.n_class], stddev)
        bias = bias_variable([self.n_class])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map
        output_map = self._reshape_to_5dim(output_map, batch_size, steps)

        return output_map, int(in_size - size)

    def _get_accuracy(self, logits, labels):
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])
        class_accuracy_list = []# = tf.zeros(shape=[self.n_class],dtype=dtypes.float32)
        labels_split = tf.split(flat_labels,self.n_class,axis=1)
        correct_prediction = tf.cast(tf.equal(tf.argmax(flat_logits, axis=1), tf.argmax(flat_labels, axis=1)), tf.float32)
        for i_class in range(self.n_class):
            labels_split[i_class] = tf.reshape(labels_split[i_class], [-1])
            i_class_correct = tf.cast(tf.tensordot(labels_split[i_class], correct_prediction, axes=1), tf.float32)
            i_class_times = tf.cast(tf.reduce_sum(labels_split[i_class]), tf.float32)
            i_class_accuracy = i_class_correct / i_class_times
            class_accuracy_list.append(tf.reshape(i_class_accuracy, [1]))
        class_accuracy = tf.concat(class_accuracy_list,axis = 0)
        total_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return total_accuracy, class_accuracy

    def _get_cross_entropy_cost(self, logits, labels):
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_labels))
        return loss

    def _get_cost(self, logits, labels, marks, cost_name, cost_kwargs = {}):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])
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
                lossmap = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_labels)
                if self.use_mark:
                    flat_marks = tf.reshape(marks, [-1])
                    lossmap = tf.tensordot(lossmap, flat_marks, axes=1)
                loss = tf.reduce_mean(lossmap)
        elif cost_name == "cross_entropy_with_class_ave_weights":
            classes_distrib_inv = 1 / tf.reduce_sum(flat_labels, axis=0)
            classes_weights = classes_distrib_inv / tf.reduce_sum(classes_distrib_inv)
            weight_map = tf.reduce_sum(flat_labels * classes_weights, axis=1)

            loss_map = tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_logits, labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)

            loss = tf.reduce_mean(weighted_loss)


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
    gtdata_shape = np.shape(gtdata)
    batch_size, steps, nx, ny, channels = iptdata_shape
    net = Conv_Net('test_convnet', nx, ny, channels, n_class)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.keep_prob: 1.0
        }
        print(sess.run(net.predict, feed_dict=feed_dict))

    #net.predict
def test_offset():
    net = Conv_Net('testoffset1', 100, 100, 3, 2)
    print(net.offsetx, ' ',net.offsety)
    net2 = Conv_Net('testoffset1', 400, 200, 3, 2)
    print(net2.offsetx, ' ',net2.offsety)

if __name__ == '__main__':
    test_offset()