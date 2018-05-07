import tensorflow as tf

from net.rnn_cell import block_C_RNNCell

def dynamic_c_lstm(nx, ny, x, x_channels, out_channels, initstate = None):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels * 2]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    with tf.variable_scope('block_c_rnn_without_size', reuse = tf.AUTO_REUSE):
        rnnCell = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2, 
            input_shape=[nx, ny, x_channels], 
            output_channels=out_channels,
            kernel_shape=[1,1]
            )
        if initstate is not None:
            initstate_split = tf.split(initstate, 2, axis = 3)
            out, _statei = tf.nn.dynamic_rnn(
                rnnCell,
                inputs=x,
                initial_state=tf.contrib.rnn.LSTMStateTuple(initstate_split[0], initstate_split[1]),
                time_major=False
            )
        else:
            out, _statei = tf.nn.dynamic_rnn(
                rnnCell,
                inputs=x,
                dtype=tf.float32,
                time_major=False
            )
    return out, rnnCell.variables

def dynamic_c_rnn(nx, ny, x, x_channels, initstate, out_state_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """

    variables = []
    with tf.variable_scope('block_c_rnn_without_size', reuse = tf.AUTO_REUSE):
        x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1], nx*ny*x_channels])
        initstate = tf.reshape(initstate, shape=[tf.shape(x)[0], nx*ny*out_state_channels], name='initstate_reshape')
        rnnCell = block_C_RNNCell(nx,ny,x_channels,out_state_channels)
        variables = variables + rnnCell.vars
        out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            #dtype = tf.float32,
            initial_state=initstate,
            time_major=False
        )
        out = tf.reshape(out, shape=[tf.shape(x)[0], tf.shape(x)[1], nx, ny, out_state_channels], name = 'out')
    return out, variables


def block_rnn(in_node, batch_size, steps, sx, sy, in_channels, out_channels, initstate, LSTM = True):
    in_node = tf.reshape(in_node, [batch_size, steps, sx, sy, in_channels])
    if LSTM is True:
        in_node, variables = dynamic_c_lstm(sx, sy, in_node, in_channels, out_channels, initstate)
    else:
        in_node, variables = dynamic_c_rnn(sx, sy, in_node, in_channels, initstate, out_channels)
    in_node = tf.reshape(in_node, [batch_size * steps, sx, sy, out_channels])
    return in_node, variables
