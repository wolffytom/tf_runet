import tensorflow as tf
import numpy as np

import sys, os
projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if projdir not in sys.path:
    sys.path.append(projdir)

from net.layer_ops import conv_relu
from net.layer_ops import dconv_relu
from net.layer_ops import fc_relu
from net.layer_ops import max_pool
from net.layer_ops import crop_and_concat
from net.block_crnn import block_rnn
from net.block_crnn import rnn

def conv_fc(in_node, keep_prob, INIT, LSTM, channels, nx, ny):
    vsname = 'conv_fc'
    if INIT:
        vsname = 'conv_fc_init'
    with tf.variable_scope(vsname, reuse = tf.AUTO_REUSE, initializer = tf.orthogonal_initializer()):
        if INIT:
            channels = channels+1
        batch_size = tf.shape(in_node)[0]
        steps = tf.shape(in_node)[1]
        in_node = tf.reshape(in_node, shape=[batch_size*steps, nx, ny, channels])
        in_node = conv_relu('conv1', in_node, 3, channels, channels, keep_prob) #128
        nx = nx-2
        ny = ny-2
        in_node = max_pool(in_node, 4) #32*32
        nx = nx//4
        ny = ny//4
        in_node = tf.layers.batch_normalization(in_node)
        in_node = conv_relu('conv4', in_node, 3, channels, channels*2, keep_prob)
        nx = nx-2
        ny = ny-2
        channels *= 2
        in_node = max_pool(in_node, 4) #8*8
        nx = nx//4
        ny = ny//4
        in_node = tf.layers.batch_normalization(in_node)
        in_node = tf.layers.flatten(in_node)
        channels = channels*nx*ny
        in_node = fc_relu('fc1', in_node, channels, 32)
        output_bands = 8
        if INIT and LSTM:
            output_bands *= 2
        in_node = fc_relu('fc2', in_node, 32, output_bands)
        in_node = tf.layers.batch_normalization(in_node)
        return in_node

def expand_to_img_size(in_node, img_sizex, img_sizey):
    in_node = tf.expand_dims(in_node, axis=1)
    in_node = tf.expand_dims(in_node, axis=1)
    return tf.tile(in_node, [1, img_sizex, img_sizey, 1])

def create_ru_net_sp_init(nx, ny, firstframe, firstlabel, otherframes, channels, keep_prob, cfg):
    layers = cfg['layers']
    features_root = cfg['features_root']
    filter_size = cfg['cnn_kernel_size']
    pool_size = cfg['pool_size']
    LSTM = cfg['LSTM']
    first_frameandlabel = tf.concat([firstframe, tf.expand_dims(firstlabel, 4)], axis=4)
    # first.shape is [batch_size, 1, nx, ny, self.channels + 1]
    initstates = {}
    variables = []
    convs = []
    pools = {}
    deconv = {}
    dw_h_convs = {}
    up_h_convs = {}


    def _ru_part(INIT, in_node):
        batch_size = tf.shape(in_node)[0]
        steps = tf.shape(in_node)[1]

        fc_res = conv_fc(in_node, keep_prob, INIT, LSTM, channels, nx, ny)
        if INIT:
            initstates['fc'] = fc_res
        else:
            with tf.variable_scope('fc_lstm', reuse = tf.AUTO_REUSE, initializer = tf.orthogonal_initializer()):
                fc_res = rnn(fc_res, batch_size, steps, 8, 8, initstates['fc'])
            fc_res = tf.reshape(fc_res, shape=[batch_size*steps, 8])

        sx = nx
        sy = ny
        in_node_channels = -1

        def block_rnn_part_different_out_channels(name_crnn, in_part, in_part_channels, out_part_channels):
            print ('---brnn_part')
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
            print ('---down:',layer)
            with tf.variable_scope('down_layer-'+str(layer)):
                features = 2**layer*features_root
                if LSTM is True and INIT is True:
                    features = features * 2
                stddev = np.sqrt(2 / (filter_size**2 * features))

                # block_rnn for input
                if layer == 0:
                    # in_node.shape is [batch_size, steps, sx, sy, ?]
                    with tf.variable_scope('input_layer', reuse=tf.AUTO_REUSE):
                        in_node_ori_channels = (channels + 1) if INIT is True else channels
                        if LSTM is True:
                            in_node_channels = 2 * (channels + 1) if INIT is True else (channels + 1)
                        else:
                            in_node_channels = channels + 1

                        in_node = tf.reshape(in_node, [-1, in_node_ori_channels])
                        in_node = fc_relu('fc_init', in_node, in_node_ori_channels, in_node_channels, stddev)
                        in_node = tf.reshape(in_node, [batch_size*steps, sx, sy, in_node_channels])
                else:
                    in_node_channels = features //2
                
                ###in_node = block_rnn_part('down'+str(layer)+'_crnn0', in_node, in_node_channels)
                in_node = conv_relu('conv1', in_node, filter_size, in_node_channels, features, keep_prob, stddev)
                sx -= 2
                sy -= 2
                ###in_node = block_rnn_part('down'+str(layer)+'_crnn1', in_node, features)
                in_node = conv_relu('conv2', in_node, filter_size, features, features, keep_prob, stddev)
                sx -= 2
                sy -= 2
                ###in_node = block_rnn_part('down'+str(layer)+'_crnn2', in_node, features)

                in_node = tf.layers.batch_normalization(in_node)
        
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
            print ('---up:',layer)
            with tf.variable_scope('up-'+str(layer), reuse = tf.AUTO_REUSE):
                features = 2**(layer+1)*features_root
                if LSTM is True and INIT is True:
                    features = features * 2
                stddev = np.sqrt(2 / (filter_size**2 * features))

                in_node = block_rnn_part('up'+str(layer)+'_in', in_node, features)

                in_node = dconv_relu('dconv', in_node, pool_size, features, features//2, stddev)
                sx *= 2
                sy *= 2
                in_node = crop_and_concat(dw_h_convs[layer], in_node)

                if layer == 0:
                    fc_expand = expand_to_img_size(fc_res, sx, sy)
                    features += 8
                    if INIT and LSTM:
                        features += 8
                    in_node =  tf.concat([in_node, fc_expand], 3)

                in_node = conv_relu('conv1', in_node, filter_size, features, features//2, keep_prob, stddev)
                sx -= 2
                sy -= 2
                in_node = block_rnn_part('up'+str(layer)+'_crnn1', in_node, features//2)
                in_node = conv_relu('conv2', in_node, filter_size, features//2, features//2, keep_prob, stddev)
                sx -= 2
                sy -= 2
                in_node = block_rnn_part('up'+str(layer)+'_crnn2', in_node, features//2)

                in_node = tf.layers.batch_normalization(in_node)
    
                up_h_convs[layer] = in_node

        if layers == 1:
            features = features_root * 2
            stddev = np.sqrt(2 / (filter_size**2 * features))
            if LSTM is True and INIT is True:
                features = features * 2

            fc_expand = expand_to_img_size(fc_res, sx, sy)
            features += 16
            if INIT and LSTM:
                features += 16
            in_node =  tf.concat([in_node, fc_expand], 3)

            in_node = conv_relu('one_layer_conv1', in_node, filter_size, features//2, features//2, keep_prob, stddev)
            sx -= 2
            sy -= 2
            in_node = conv_relu('one_layer_conv2', in_node, filter_size, features//2, features//2, keep_prob, stddev)
            sx -= 2
            sy -= 2
    
        # last block_rnn
        ### in_node = block_rnn_part('last_block_rnn', in_node, features//2)

        # returns of this part func
        if True == INIT:
            return None
        else:
            output_map = conv_relu('conv_out', in_node, 1, in_node.shape[3], 1, 1.0, stddev, relu_=False)
            up_h_convs["out"] = output_map

            output_map = tf.reshape(output_map, [batch_size, steps, sx, sy])

            # softmax
            output_map = tf.nn.sigmoid(output_map, name = 'output_map')
            return output_map

    # calculate the initstates
    with tf.variable_scope('init_frame', reuse = tf.AUTO_REUSE):
        in_node = first_frameandlabel
        in_node = tf.layers.batch_normalization(in_node)
        _ru_part(True, in_node)

    # process other frames
    with tf.variable_scope('other_frames', reuse = tf.AUTO_REUSE):
        in_node = otherframes
        in_node = tf.layers.batch_normalization(in_node)
        return _ru_part(False, in_node)

def calculate_offset(nx, ny, cfg):
    sx = nx
    sy = ny
    layers = cfg['layers']
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
        sx -= 4 
        sy -= 4 
    offsetx = (nx - sx) // 2
    offsety = (ny - sy) // 2
    return sx, offsetx, sy, offsety

if __name__ == '__main__':
    from config import cfg
    batch_size = 10
    othersteps = 20
    bands = 3
    nx = 100
    ny = 100
    keep_prob = 0.9
    firstframe = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, 1, nx, ny, bands])
    firstlabel = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, 1, nx, ny])
    otherframes = tf.constant(1.0, dtype = tf.float32, shape=[batch_size, othersteps, nx, ny, bands])
    res = create_ru_net_sp_init(nx, ny, firstframe, firstlabel, otherframes, bands, keep_prob, cfg)
