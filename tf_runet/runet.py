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
from PIL import Image

import tensorflow as tf
from config import *

from conv_network import Conv_Net
from vot2016 import VOT2016_Data_Provider
from util import crop_video_to_shape_with_offset
from util import oneHot_to_gray255

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class RUNet(object):
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
        self.optimizer = self._create_optimizer()
        self._create_global_net(nx=global_nx, ny=global_ny)
        self.global_offset = self.global_net.offsetx
        
        self.cost = tf.placeholder(name = 'runet.cost', dtype = tf.float32, shape=None)
        tf.summary.scalar('cost', self.cost)
        self.total_accuracy = tf.placeholder(dtype = tf.float32, shape=None)
        tf.summary.scalar('total_accuracy', self.total_accuracy)
        self.class_accuracy = tf.placeholder(dtype = tf.float32, shape=[self.n_class])
        class_accuracy_list = tf.split(self.class_accuracy, self.n_class,axis=0)
        self.predict = tf.placeholder(dtype = tf.float32, shape=[None, None, None, None, self.n_class])
        predict_flat = tf.reshape(self.predict, [-1, self.n_class])
        predict_flat_split = tf.split(predict_flat, self.n_class,axis=1)
        for i in range(self.n_class):
            tf.summary.scalar('class_' + str(i) + '_accuracy', tf.reshape(class_accuracy_list[i], shape = []))
            tf.summary.histogram('class_'+ str(i) + '_predict', tf.reshape(predict_flat_split[i], [-1]))
    
    def _init_vars_random(self):
        self.sess.run(tf.global_variables_initializer())
    
    def _create_global_net(self, opt_kwargs={}, nx=100, ny=100):
        self.global_net = Conv_Net('global_net', nx, ny, self.channels, self.n_class)
        self.global_minimizer = self.optimizer.minimize(self.global_net.cost)
    
    def _create_net(self, name, nx, ny, opt_kwargs={}):
        net = Conv_Net(name, nx, ny, self.channels, self.n_class)
        return net

    def _create_optimizer(self, optimizer = "Adam"):
        with tf.variable_scope('runet.optimizer'):
            if optimizer == "RMSProp":
                return tf.train.RMSPropOptimizer(learning_rate = args.learning_rate)
            elif optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            else:
                return None
    
    def predict(self, iptdata, gtdata):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert self.channels == channels

        netname = 'predict_net'
        net = self._create_net(netname, nx, ny)
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.keep_prob: 1.0
        }
        cost, accuracy, otherlabels, predict = self.sess.run((
            net.cost,
            net.accuracy, 
            net.otherlabels,
            net.predict), feed_dict=feed_dict)
        print ('cost as:' , cost)
        return cost, accuracy, otherlabels, predict

    def train(self, iptdata, gtdata, get_mark_func, optimizer="momentum", opt_kwargs={}):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert self.channels == channels

        netname = 'train_net'
        net = self._create_net(netname, nx, ny)
        feed_dict = {
            net.inputs: iptdata,
            net.labels: gtdata,
            net.othermarks: get_mark_func((iptdata, gtdata),net.offsetx,(net.sx,net.sy)),
            net.keep_prob: 0.5
        }
        _opt, cost, total_accuracy, class_accuracy, otherlabels, predict = self.sess.run((
            self.optimizer.minimize(net.cost),
            net.cost,
            net.total_accuracy,
            net.class_accuracy,
            net.otherlabels,
            net.predict), feed_dict=feed_dict)
        
        summary = self.sess.run(tf.summary.merge_all(), feed_dict={
            self.cost:cost, 
            self.total_accuracy:total_accuracy,
            self.class_accuracy:class_accuracy,
            self.predict:predict})
        return summary, cost, total_accuracy, class_accuracy, otherlabels, predict

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

    def restore(self, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
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