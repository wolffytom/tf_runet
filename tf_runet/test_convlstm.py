import tensorflow as tf
from tensorflow.contrib.rnn import ConvLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

def block_c_rnn_ConvLSTMCell(nx, ny, x, x_channels, initstate, out_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels * 2]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    with tf.variable_scope('block_c_rnn_without_size', reuse = tf.AUTO_REUSE):
        initstate_split = tf.split(initstate, 2, axis = 3)
        rnnCell = ConvLSTMCell(
            conv_ndims=2, 
            input_shape=[nx, ny, x_channels], 
            output_channels=out_channels,
            kernel_shape=[1,1]
            )
        out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            initial_state=LSTMStateTuple(initstate_split[0], initstate_split[1]),
            time_major=False
        )
    return out, rnnCell.variables

def block_c_rnn_with_size(batch_size, steps, nx, ny, x, x_channels, initstate, out_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels * 2]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    with tf.variable_scope('block_c_rnn_without_size', reuse = tf.AUTO_REUSE):
        initstate_split = tf.split(initstate, 2, axis = 3)
        rnnCell = ConvLSTMCell(
            conv_ndims=2, 
            input_shape=[7,8,3], 
            output_channels=out_channels,
            kernel_shape=[1,1]
        )
        out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            #dtype = tf.float32,
            initial_state=LSTMStateTuple(initstate_split[0], initstate_split[1])
        )
    print('out:',out)
    print('_statei:', _statei)
    #out = tf.reshape(out, shape=[batch_size, steps, nx, ny, out_channels])
    return out


x_ipt = tf.placeholder(tf.float32, shape=[5,6,7,8,3])
init_state = tf.placeholder(tf.float32, shape=[5,7,8,8])
#o = block_c_rnn_with_size(5,6,7,8,x_ipt,3,init_state,4)
o = block_c_rnn_without_size(7,8,x_ipt,3,init_state,4)
