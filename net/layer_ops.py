import tensorflow as tf

def weight_variable(name, shape, stddev=0.1):
    return tf.get_variable(name, shape=shape, initializer=tf.orthogonal_initializer())

def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.zeros_initializer())

def fc_relu(name, x, in_channels, out_channels, w_stddev=1.):
    with tf.variable_scope(name):
        w = weight_variable('w', [in_channels, out_channels], w_stddev)
        b = bias_variable('b', [out_channels])
        return tf.nn.relu(tf.matmul(x, w) + b)

def conv_relu(name, x, filter_size, in_channels, out_channels, keep_prob, w_stddev=1., relu_=True):
    with tf.variable_scope(name, initializer = tf.random_normal_initializer()):
        w = weight_variable('w', [filter_size, filter_size, in_channels, out_channels], w_stddev)
        b = bias_variable('b', [out_channels])
        res = tf.nn.dropout(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID'), keep_prob)
        if relu_:
            res = tf.nn.relu(res)
        return res

def dconv_relu(name, x, pool_size, in_channels, out_channels, w_stddev):
    with tf.variable_scope(name):
        w = weight_variable('wd', [pool_size, pool_size, out_channels, in_channels], w_stddev)
        b = bias_variable('bd', [out_channels])

        dconv = deconv2d(x, w, pool_size, name=name)
        return tf.nn.relu(dconv+b)

def conv2d(x, W, keep_prob_, name=None):
    if name is None:
        name = 'conv2d_noname'
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        return tf.nn.dropout(conv_2d, keep_prob_, name=name)

def deconv2d(x, W,stride, name=None):
    if name is None:
        name = 'deconv2d'
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name=name)

def max_pool(x,n, name=None):
    if name is None:
        name = 'max_pool'
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID', name=name)

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)


