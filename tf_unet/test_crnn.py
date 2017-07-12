from runet import block_rnn_zeroinit
from crnn import c_rnn

import tensorflow as tf

steps = 20
nx = 100
ny = 100
channals = 3
n_class = 2
x = tf.placeholder(tf.float32, shape=(None, nx, ny, channals))
gt1 = tf.placeholder(tf.float32, shape=(1, nx, ny, n_class))
#create_r_conv_net_nobatch(x, gt1, True, channals, n_class)
#out = block_rnn_zeroinit(x,nx,ny,channals)
out = c_rnn(x)

print (out)
print ('helloworld')