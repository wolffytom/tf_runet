from tensorflow.contrib import rnn
from layers import weight_variable
from layers import bias_variable
from layers import conv2d

import tensorflow as tf

def block_c_rnn(x):
    """c-rnn

    Args:
        x: inputs, size as [steps, nx, ny, channels]
    """
    x_shapelist = x.get_shape().as_list()
    nx = x_shapelist[1]
    ny = x_shapelist[2]
    channels = x_shapelist[3]

    x = tf.reshape(x, shape=[1, -1, nx*ny*channels])
    initstate = tf.constant(0, dtype=tf.float32, shape=[1, nx* ny* channels])
    rnnCell = block_C_RNNCell(nx,ny,channels)
    out, _statei = tf.nn.dynamic_rnn(
            rnnCell,
            inputs=x,
            initial_state=initstate,
            time_major=False
        )
    out = tf.reshape(out, shape=[-1, nx, ny, channels])
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

    def __init__(self, nx, ny, channels, keep_prob=1, reuse=None):
        super(block_C_RNNCell, self).__init__(_reuse=reuse)
        self.nx = nx
        self.ny = ny
        self.channels = channels
        #self._activation = activation or math_ops.tanh
        self.keep_prob = keep_prob
        with tf.variable_scope('block_C_RNNCell') as vs:
            self._weight = weight_variable([1, 1, channels * 2, channels])
            self._bias = bias_variable([channels])

    @property
    def state_size(self):
        return self.nx * self.ny * self.channels

    @property
    def output_size(self):
        return self.nx * self.ny * self.channels

    def call(self, inputs, state, scope='block_R_CNNCell'):
        """
        call func.

        :param inputs: input tensor, shape [1(batch_size),nx , ny , channels]
        :param state: last state, shape [1(batch_size),nx , ny , channels]
        """
        
        #input_concat = tf.concat([inputs, state], axis=3)
        output = inputs#tf.nn.relu(self._bias + conv2d(input_concat, self._weight, self.keep_prob))
        return output, output