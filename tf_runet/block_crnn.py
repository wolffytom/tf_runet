from tensorflow.contrib import rnn
from layers import weight_variable
from layers import bias_variable
from layers import conv2d

import tensorflow as tf

def block_c_rnn_without_size(nx, ny, x, x_channels, initstate, out_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    x_shapelist = x.get_shape().as_list()
    assert x_shapelist[2] == nx and x_shapelist[3] == ny and x_shapelist[4] == x_channels and len(x_shapelist) == 5
    initstate_shapelist = initstate.get_shape().as_list()
    assert initstate_shapelist[1] == nx and initstate_shapelist[2] == ny and initstate_shapelist[3] == out_channels and len(initstate_shapelist) == 4

    x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1], nx*ny*x_channels])
    initstate = tf.reshape(initstate, shape=[tf.shape(x)[0], nx*ny*out_channels])
    rnnCell = block_C_RNNCell(nx,ny,x_channels,out_channels)
    out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            #dtype = tf.float32,
            initial_state=initstate,
            time_major=False
        )
    out = tf.reshape(out, shape=[tf.shape(x)[0], tf.shape(x)[1], nx, ny, out_channels])
    return out

def block_c_rnn_zero_init_without_size(nx, ny, x, x_channels, out_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, x_channels]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    x_shapelist = x.get_shape().as_list()
    assert x_shapelist[2] == nx and x_shapelist[3] == ny and x_shapelist[4] == x_channels and len(x_shapelist) == 5
    #initstate_shapelist = initstate.get_shape().as_list()
    #assert initstate_shapelist[1] == nx and initstate_shapelist[2] == ny and initstate_shapelist[3] == out_channels and len(initstate_shapelist) == 4

    x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1], nx*ny*x_channels])
    #initstate = tf.reshape(initstate, shape=[tf.shape(x)[0], nx*ny*out_channels])
    rnnCell = block_C_RNNCell(nx,ny,x_channels,out_channels)
    out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            dtype = tf.float32,
            #initial_state=initstate,
            time_major=False
        )
    out = tf.reshape(out, shape=[tf.shape(x)[0], tf.shape(x)[1], nx, ny, out_channels])
    return out

def block_c_rnn_zero_init_with_size(batch_size, steps, nx, ny, x, x_channels, out_channels):
    initstate = tf.constant(0, dtype=tf.float32, shape=[batch_size, nx, ny, out_channels])
    return block_c_rnn(batch_size, steps, nx, ny, x, x_channels, initstate, out_channels)

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
    initstate_shapelist = initstate.get_shape().as_list()
    assert initstate_shapelist == [batch_size, nx, ny, out_channels]

    x = tf.reshape(x, shape=[batch_size, steps, nx*ny*x_channels])
    initstate = tf.reshape(initstate, shape=[batch_size, nx*ny*out_channels])
    rnnCell = block_C_RNNCell(nx,ny,x_channels,out_channels)
    out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            #dtype = tf.float32,
            initial_state=initstate,
            time_major=False
        )
    out = tf.reshape(out, shape=[batch_size, steps, nx, ny, out_channels])
    return out
    

class block_C_RNNCell(rnn.RNNCell):
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the LSTM cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """

    def __init__(self, nx, ny, in_channels, out_channels, keep_prob=1, reuse=None):
        super(block_C_RNNCell, self).__init__(_reuse=reuse)
        self.nx = nx
        self.ny = ny
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.keep_prob = keep_prob
        with tf.variable_scope('block_C_RNNCell') as vs:
            self._weight = weight_variable([1, 1, in_channels + out_channels, out_channels])
            self._bias = bias_variable([out_channels])

    @property
    def state_size(self):
        return self.nx * self.ny * self.out_channels

    @property
    def output_size(self):
        return self.nx * self.ny * self.out_channels

    def call(self, inputs, state, scope='block_R_CNNCell'):
        """
        call func.

        :param inputs: input tensor, shape [batch_size ,nx , ny , channels]
        :param state: last state, shape [batch_size ,nx , ny , channels]
        """
        
        inputs = tf.reshape(inputs, shape=[-1, self.nx, self.ny, self.in_channels])
        state = tf.reshape(state, shape=[-1, self.nx, self.ny, self.out_channels])
        input_concat = tf.concat([inputs, state], axis=3)
        output = tf.nn.relu(self._bias + conv2d(input_concat, self._weight, self.keep_prob))
        output = tf.reshape(output, shape=[-1, self.nx*self.ny*self.out_channels])
        return output, output

def test_with_size():
    batch_size = 100
    steps = 20
    nx = 100
    ny = 100
    channels = 3
    n_class = 2
    x = tf.placeholder(tf.float32, shape=(batch_size, steps, nx, ny, channels))
    initstate = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, n_class))
    #create_r_conv_net_nobatch(x, gt1, True, channals, n_class)
    #out = block_rnn_zeroinit(x,nx,ny,channals)
    out = block_c_rnn_zero_init(batch_size, steps, nx, ny, x, channels, n_class)

    print (out)
    print ('helloworld')

def test_without_size():
    batch_size = 100
    steps = 20
    nx = 100
    ny = 100
    channels = 3
    n_class = 2
    x = tf.placeholder(tf.float32, shape=(None, None, nx, ny, channels))
    initstate = tf.placeholder(tf.float32, shape=(None, nx, ny, n_class))
    #create_r_conv_net_nobatch(x, gt1, True, channals, n_class)
    #out = block_rnn_zeroinit(x,nx,ny,channals)
    out = block_c_rnn_without_size(nx, ny, x, channels, initstate, n_class)

    print (out)
    print ('helloworld')

test_without_size()