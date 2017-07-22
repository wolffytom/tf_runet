import tensorflow as tf

import numpy as np

class BasicACNetwork(object):
    def __init__(self,name):
        self._name = "ACnet_" + str(name)

    def sync_from(self, src_netowrk, name=None):
        '''
        return a list of ops
        run the list will sync self from src_network
        '''
        src_vars = src_netowrk.vars
        dst_vars = self.vars

        sync_ops = []

        with tf.name_scope(name, "ACNetwork", []) as name:
            for(src_var, dst_var) in zip(src_vars, dst_vars):
                sync_op = tf.assign(dst_var, src_var)
                sync_ops.append(sync_op)

            return tf.group(*sync_ops, name=name)

    def _fc_variable(self, weight_shape):
        d = 1.0 / np.sqrt(weight_shape[0])
        bias_shape = [weight_shape[1]]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name='weights')
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d), name='bias')
        return weight, bias