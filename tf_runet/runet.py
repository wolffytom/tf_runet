# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
import time
import util

import tensorflow as tf

from conv_network import Conv_Net
from vot2016 import VOT2016_Data_Provider
from util import crop_video_to_shape_with_offset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class RUnet_test(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, name, global_nx = 100, global_ny = 100, channels=3, n_class=2, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()
        
        self.name = name
        self.channels = channels
        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)

        self.sess = tf.Session()
        self.global_net_idx = 0
        self.global_step = tf.Variable(0)
        self._create_global_net_and_init_vars(nx=global_nx, ny=global_ny)
        """
        self.x = tf.placeholder("float", shape=[None, None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        logits, self.variables, self.offset = create_conv_r_net_test(
            self.x , self.keep_prob, channels, n_class, **kwargs)

        self.cost = self._get_cost(logits, cost, cost_kwargs)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))

        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(
            tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        """
    
    def _create_global_net_and_init_vars(self, opt_kwargs={}, nx=100, ny=100):
        self.global_net = Conv_Net(self.name + '.' + str(self.global_net_idx) + '.global_net', nx, ny, self.channels, self.n_class)
        self.global_net_idx = self.global_net_idx + 1
        self.global_net.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,**opt_kwargs).minimize(self.global_net.cost)
        self.global_net.refresh_variables()

        #for v in (self.global_net.optimizer._g):
        #    print(v, self.sess.run(v))
        self.sess.run(tf.global_variables_initializer())
    
    def _create_net_and_copy_vars(self, name, nx, ny, opt_kwargs={}):
        net = Conv_Net(self.name + '.' + str(self.global_net_idx) + '.' + name, nx, ny, self.channels, self.n_class)
        self.global_net_idx = self.global_net_idx + 1
        #net.optimizer = tf.train.AdamOptimizer(learning_rate=0.001,**opt_kwargs).minimize(net.cost,
        #                                                                   global_step=self.global_step)
        net.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,**opt_kwargs).minimize(net.cost)
        net.refresh_variables()
        self.sess.run(net.sync_from(self.global_net))
        return net

    def _refresh_global_vars(self, net):
        self.sess.run(self.global_net.sync_from(net))
    
    def get_predict_results(self, iptdata, gtdata):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert self.channels == channels
        assert np.shape(gtdata) == (batch_size, steps, nx, ny, self.n_class)

        netname = 'predict_net' + '_as_size_nx' + str(nx) + '_ny' + str(ny)
        net = self._create_net_and_copy_vars(netname, nx, ny)
        gtdata = crop_video_to_shape_with_offset(gtdata, net.offset)
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.keep_prob: 1.
        }
        results = self.sess.run(net.predict, feed_dict=feed_dict)
        return results
    
    def get_predict_cost(self, iptdata, gtdata):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert self.channels == channels
        assert np.shape(gtdata) == (batch_size, steps, nx, ny, self.n_class)

        netname = 'predict_net' + '_as_size_nx' + str(nx) + '_ny' + str(ny)
        net = self._create_net_and_copy_vars(netname, nx, ny)
        gtdata = crop_video_to_shape_with_offset(gtdata, net.offset)
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.keep_prob: 1.
        }
        cost = self.sess.run(net.cost, feed_dict=feed_dict)
        return cost

    def train_globalnet(self, iptdata, gtdata, optimizer="momentum", opt_kwargs={}):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert self.channels == channels

        netname = 'train_net'# + '_as_size_nx' + str(nx) + '_ny' + str(ny)
        net = self.global_net#self._create_net_and_copy_vars(netname, nx, ny)
        gtdata = crop_video_to_shape_with_offset(gtdata, net.offset)
        learning_rate = opt_kwargs.pop("learning_rate", 0.2)
        decay_rate = opt_kwargs.pop("decay_rate", 0.95)
        momentum = opt_kwargs.pop("momentum", 0.2)
        rmsp_epsilon = opt_kwargs.pop("rmsp_epsilon", 0.1)
        grad_applier = tf.train.RMSPropOptimizer(
            learning_rate = learning_rate,
            decay = decay_rate,
            momentum = momentum, 
            epsilon = rmsp_epsilon)
        #for v in (tf.global_variables()):
        #    print(v)
        #    print(self.sess.run(v))
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.keep_prob: 1.0
        }
        _opt, cost = self.sess.run((net.optimizer,net.cost), feed_dict=feed_dict)
        print ('cost as:' , cost)
        #self._refresh_global_vars(net)
        return cost

    def train(self, iptdata, gtdata, optimizer="momentum", opt_kwargs={}):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert self.channels == channels

        netname = 'train_net'# + '_as_size_nx' + str(nx) + '_ny' + str(ny)
        net = self._create_net_and_copy_vars(netname, nx, ny)
        gtdata = crop_video_to_shape_with_offset(gtdata, net.offset)
        learning_rate = opt_kwargs.pop("learning_rate", 0.2)
        decay_rate = opt_kwargs.pop("decay_rate", 0.95)
        momentum = opt_kwargs.pop("momentum", 0.2)
        rmsp_epsilon = opt_kwargs.pop("rmsp_epsilon", 0.1)
        grad_applier = tf.train.RMSPropOptimizer(
            learning_rate = learning_rate,
            decay = decay_rate,
            momentum = momentum, 
            epsilon = rmsp_epsilon)
        #for v in (tf.global_variables()):
        #    print(v)
        #    print(self.sess.run(v))
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.keep_prob: 1.0
        }
        _opt, cost = self.sess.run((net.optimizer,net.cost), feed_dict=feed_dict)
        print ('cost as:' , cost)
        self._refresh_global_vars(net)
        return cost

    def save(self, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(self.sess, model_path)
        print("Model saved in file: %s" % save_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

def test_train():
    from PIL import Image
    print('begin')
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    iptdata, gtdata = dptest.get_data_one_batch(8)
    iptdata = iptdata[:,0:1,:,:,:]
    gtdata = gtdata[:,0:1,:,:,:]

    runet = RUnet_test('runet_test', global_nx = np.shape(iptdata)[2], global_ny = np.shape(iptdata)[3])

    for i in range(100):
        cost = runet.train_globalnet(iptdata, gtdata)
        #predictresult = runet.get_predict_results(iptdata, gtdata)
        print(i,'-------',cost,'\n')
        #if (i % 10 == 0):
            #savename = '/home/cjl/model/20170723tf' + str(i)
            #runet.save(savename)
        #if (i % 20 == 0):
            #(Image.fromarray(util.oneHot_to_gray255(predictresult[0][5]))).show(title='0,5')
    print('========================================')
    runet.save('/home/cjl/model/20170723tf')
    #cost = runet.train(iptdata, gtdata)
    print(cost)

def test_predict_results():
    from PIL import Image
    runet = RUnet_test('runet_test')
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    iptdata, gtdata = dptest.get_data_one_batch(8)
    results = runet.get_predict_results(iptdata, gtdata)
    results = util.oneHot_to_gray255(results[0][5])
    im_t_PIL = Image.fromarray(results)
    im_t_PIL.show()
    print(results)
    print(type(results))



if __name__ == '__main__':
    test_train()