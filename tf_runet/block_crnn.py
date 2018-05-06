from tensorflow.contrib import rnn
from layers import weight_variable
from layers import bias_variable
from layers import conv2d

import tensorflow as tf

from tensorflow.contrib.rnn import ConvLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

def block_c_lstmnn(nx, ny, x, x_channels, out_channels, initstate = None):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, channels]
        initstate: size as [batch_size, nx, ny, out_channels * 2]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """
    with tf.variable_scope('block_c_rnn_without_size', reuse = tf.AUTO_REUSE):
        rnnCell = ConvLSTMCell(
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
                initial_state=LSTMStateTuple(initstate_split[0], initstate_split[1]),
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

def block_c_rnn(nx, ny, x, x_channels, initstate, out_state_channels):
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

def block_c_rnn_zero_init(nx, ny, x, x_channels, out_channels):
    """c-rnn

    Args:
        x: inputs, size as [batch_size, steps, nx, ny, x_channels]
    
    return:
        output: size as [batch_size, steps, nx, ny, out_channels]
    """

    x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1], nx*ny*x_channels])
    rnnCell = block_C_RNNCell(nx,ny,x_channels,out_channels)
    out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            dtype = tf.float32,
            time_major=False
        )
    out = tf.reshape(out, shape=[tf.shape(x)[0], tf.shape(x)[1], nx, ny, out_channels])
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
        self.vars = []
        with tf.variable_scope('block_C_RNNCell', reuse = tf.AUTO_REUSE) as vs:
            self._weight = weight_variable('block_C_RNNCell_w1', [1, 1, in_channels + out_channels, out_channels])
            self._bias = bias_variable('block_C_RNNCell_b1', [out_channels])
            self.vars.append(self._weight)
            self.vars.append(self._bias)

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
    out, variables = block_c_rnn(nx, ny, x, channels, initstate, n_class)

    print ("out:",out)
    print ("vars:",variables)
    print ('helloworld')

if __name__ == '__main__':
    test_without_size()
