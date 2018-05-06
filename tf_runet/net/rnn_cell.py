import tensorflow as tf

from tensorflow.contrib import rnn
from layer_ops import weight_variable
from layer_ops import bias_variable
from layer_ops import conv2d

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
