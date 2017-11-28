import tensorflow as tf
from tensorflow.contrib.rnn import ConvLSTMCell
def block_c_rnn_with_size(batch_size, steps, nx, ny, x, x_channels, initstate, out_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    x_shapelist = x.get_shape().as_list()
    assert x_shapelist == [batch_size, steps, nx, ny, x_channels]
    #initstate_shapelist = initstate.get_shape().as_list()
    #assert initstate_shapelist == [batch_size, nx, ny, out_channels]

    #x = tf.reshape(x, shape=[batch_size, steps, nx*ny*x_channels])
    #initstate = tf.reshape(initstate, shape=[batch_size, nx*ny*out_channels])
    rnnCell = ConvLSTMCell(
        conv_ndims=2, 
        input_shape=[7,8,3], 
        output_channels=out_channels,
        kernel_shape=[3,3]
    )
    #rnnCell = block_C_RNNCell(nx,ny,x_channels,out_channels)
    out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            dtype = tf.float32,
            #initial_state=initstate,
        )
    print('out:',out)
    print('_statei:', _statei)
    #out = tf.reshape(out, shape=[batch_size, steps, nx, ny, out_channels])
    return out

x_ipt = tf.placeholder(tf.float32, shape=[5,6,7,8,3])
o = block_c_rnn_with_size(5,6,7,8,x_ipt,3,None,4)