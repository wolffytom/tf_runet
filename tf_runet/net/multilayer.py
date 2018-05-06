import tensorflow as tf
import numpy as np
from collections import OrderedDict

from layer_ops import conv_relu
from layer_ops import dconv_relu

def create_ru_net_sp_init(nx, ny, firstframe, firstlabel, otherframes, channels, n_class,
        layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True, LSTM = True):
    first_frameandlabel = tf.concat([firstframe,firstlabel], axis = 4)
    # first.shape is [batch_size, 1, nx, ny, self.channels + self.n_class]
    initstates = {}
    variables = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()


    def _ru_part(INIT, in_node):
        batch_size = tf.shape(in_node)[0]
        steps = tf.shape(in_node)[1]

        sx = nx
        sy = ny
        in_node_channels = -1
        in_node = tf.reshape(in_node, [tf.to_int32(batch_size * steps), sx, sy, channels + n_class])

        def block_rnn_part_different_out_channels(name_crnn, in_part, in_part_channels, out_part_channels):
            if LSTM is True:
                state_channels = out_part_channels * 2
            else:
                state_channels = out_part_channels
            if INIT is True:
                #init_part.shape = [batch_size, 1, sx, sy, in_part_channels]
                initstates[name_crnn] = tf.reshape(in_part, [batch_size, sx, sy, in_part_channels])
                return in_part
            else:
                with tf.variable_scope('crnn-'+name_crnn, reuse = tf.AUTO_REUSE, initializer = orthogonal_initializer()):
                    initstate = initstates[name_crnn]
                    # initstate.shape is [batch_size, sx, sy, state_channels]
                    initstate_shape = initstate.get_shape().as_list()
                    assert len(initstate_shape) == 4 and state_channels == initstate_shape[3]
                    out_part, block_rnn_vars = self._block_rnn(in_part, batch_size, steps, sx, sy, in_part_channels, out_part_channels, initstates[name_crnn], LSTM)
                    variables.extend(block_rnn_vars)
                    return out_part
        
        def block_rnn_part(name_crnn, in_part, in_part_channels):
            return block_rnn_part_different_out_channels(name_crnn, in_part, in_part_channels, in_part_channels)
        
        # down layers
        for layer in range(0, layers):
            with tf.variable_scope('down_layer-'+str(layer)):
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
                
                in_node = block_rnn_part('down'+str(layer)+'_crnn0', in_node, in_node_channels)
                in_node = conv_relu('conv1', in_node, filter_size, in_node_channels, features, keep_prob, stddev)
                sx -= 2
                sy -= 2
                in_node = block_rnn_part('down'+str(layer)+'_crnn1', in_node, features)
                in_node = conv_relu('conv2', in_node, filter_size, features, features, keep_prob, stddev)
                sx -= 2
                sy -= 2
                in_node = block_rnn_part('down'+str(layer)+'_crnn2', in_node, features)
        
                #variables.extend((w1,b1,w2,b2))
    
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
            with tf.variable_scope('up-'+str(layer), reuse = tf.AUTO_REUSE):
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
                #h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                in_node = dconv_relu('dconv', in_node, pool_size, features, features//2, stddev)
                sx *= 2
                sy *= 2
                in_node = crop_and_concat(dw_h_convs[layer], in_node)
                #deconv[layer] = h_deconv_concat
                in_node = conv_relu('conv1', in_node, filter_size, features, features//2, keep_prob, stddev)
                sx -= 2
                sy -= 2
                in_node = block_rnn_part('up'+str(layer)+'_crnn1', in_node, features//2)
                in_node = conv_relu('conv2', in_node, filter_size, features//2, features//2, keep_prob, stddev)
                sx -= 2
                sy -= 2
                in_node = block_rnn_part('up'+str(layer)+'_crnn2', in_node, features//2)
    
                up_h_convs[layer] = in_node

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
    with tf.variable_scope('init_frame', reuse = tf.AUTO_REUSE):
        _ru_part(True, first_frameandlabel)

    # process other frames
    with tf.variable_scope('other_frames', reuse = tf.AUTO_REUSE):
        return _ru_part(False, otherframes), variables

if __name__ == '__main__':
    batch_size = 10
    othersteps = 20
    bands = 3
    class_num = 2
    nx = 100
    ny = 100
    firstframe = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, 1, nx, ny, bands])
    firstlabel = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, 1, nx, ny, class_num])
    otherframes = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, othersteps, nx, ny, bands])
    res, vs = create_ru_net_sp_init(nx, ny, firstframe, firstlabel, otherframes)
    print(res)
