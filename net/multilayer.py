import tensorflow as tf
import numpy as np
from collections import OrderedDict

from net.layer_ops import conv_relu
from net.layer_ops import dconv_relu
from net.layer_ops import fc_relu
from net.layer_ops import max_pool
from net.layer_ops import crop_and_concat
from net.block_crnn import block_rnn

def create_ru_net_sp_init(nx, ny, firstframe, firstlabel, otherframes, channels, n_class, keep_prob, cfg):
    layers = cfg.layers
    features_root = cfg.features_root
    filter_size = cfg.cnn_kernel_size
    pool_size = cfg.pool_size
    LSTM = cfg.LSTM
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
                with tf.variable_scope('crnn-'+name_crnn, reuse = tf.AUTO_REUSE, initializer = tf.orthogonal_initializer()):
                    initstate = initstates[name_crnn]
                    # initstate.shape is [batch_size, sx, sy, state_channels]
                    initstate_shape = initstate.get_shape().as_list()
                    assert len(initstate_shape) == 4 and state_channels == initstate_shape[3]
                    out_part, block_rnn_vars = block_rnn(in_part, batch_size, steps, sx, sy, in_part_channels, out_part_channels, initstates[name_crnn], LSTM)
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
                        in_node_ori_channels = (channels + n_class) if INIT is True else channels
                        if LSTM is True:
                            in_node_channels = 2 * (channels + n_class) if INIT is True else (channels + n_class)
                        else:
                            in_node_channels = channels + n_class

                        in_node = tf.reshape(in_node, [-1, in_node_ori_channels])
                        in_node = fc_relu('fc_init', in_node, in_node_ori_channels, in_node_channels, stddev)
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
                dw_h_convs[layer] = in_node
                
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

        if layers == 1:
            features = features_root * 2
            if LSTM is True and INIT is True:
                features = features * 2
            in_node = conv_relu('one_layer_last_conv', in_node, filter_size, features//2, features//2, keep_prob, stddev)
            sx -= 2
            sy -= 2
    
        # last block_rnn
        in_node = block_rnn_part('last_block_rnn', in_node, features//2)

        # returns of this part func
        if True == INIT:
            return None
        else:
            output_map = conv_relu('conv_out', in_node, 1, features_root, n_class, 1.0, stddev)
            up_h_convs["out"] = output_map

            output_map = tf.reshape(output_map, [batch_size, steps, sx, sy, n_class])

            # softmax
            output_map = tf.nn.softmax(output_map, name = 'output_map')
            return output_map

    # calculate the initstates
    with tf.variable_scope('init_frame', reuse = tf.AUTO_REUSE):
        _ru_part(True, first_frameandlabel)

    # process other frames
    with tf.variable_scope('other_frames', reuse = tf.AUTO_REUSE):
        return _ru_part(False, otherframes)

def calculate_offset(nx, ny, cfg):
    sx = nx
    sy = ny
    layers = cfg.layers
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
    if layers == 1:
        sx -= 2
        sy -= 2
    offsetx = (nx - sx) // 2
    offsety = (ny - sy) // 2
    return sx, offsetx, sy, offsety

from config import cfg
if __name__ == '__main__':
    batch_size = 10
    othersteps = 20
    bands = 3
    class_num = 2
    nx = 100
    ny = 100
    keep_prob = 0.9
    firstframe = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, 1, nx, ny, bands])
    firstlabel = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, 1, nx, ny, class_num])
    otherframes = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, othersteps, nx, ny, bands])
    res, vs = create_ru_net_sp_init(nx, ny, firstframe, firstlabel, otherframes, bands, class_num, keep_prob, cfg)
    print(res)
