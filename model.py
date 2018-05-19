import numpy as np
import tensorflow as tf

from ru_net import Ru_net
from config import cfg

class Model(object):
    def __init__(self):
        self._nets = {}
        self._optimizer = self._create_optimizer(cfg.optimizer)
        self._sess = tf.Session()
        self._base_net = self.get_net(cfg.base_net_size, cfg.base_net_size)

        self.cost = tf.placeholder(name = 'runet.cost', dtype = tf.float32, shape=None)
        tf.summary.scalar('cost', self.cost)
        self.total_accuracy = tf.placeholder(dtype = tf.float32, shape=None)
        tf.summary.scalar('total_accuracy', self.total_accuracy)
        self.class_accuracy = tf.placeholder(dtype = tf.float32, shape=[cfg.n_class])
        class_accuracy_list = tf.split(self.class_accuracy, cfg.n_class,axis=0)
        self.predict = tf.placeholder(dtype = tf.float32, shape=[None, None, None, None, cfg.n_class])
        predict_flat = tf.reshape(self.predict, [-1, cfg.n_class])
        predict_flat_split = tf.split(predict_flat, cfg.n_class,axis=1)
        for i in range(cfg.n_class):
            tf.summary.scalar('class_' + str(i) + '_accuracy', tf.reshape(class_accuracy_list[i], shape = []))
            tf.summary.histogram('class_'+ str(i) + '_predict', tf.reshape(predict_flat_split[i], [-1]))
 

    def get_net(self, nx, ny):
        scrpt = False
        netname = str(nx) + ',' + str(ny)
        if netname in self._nets:
            if scrpt:
                print('get_net_from_nets')
            return self._nets[netname]
        else:
            if scrpt:
                print('create_net')
            newnet = Ru_net(nx, ny, netname)
            self._nets[netname] = newnet
            return newnet

    def train(self, iptdata, gtdata, get_mark_func):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert cfg.channels == channels
        n_class = np.shape(gtdata)[4]
        assert cfg.n_class == n_class

        net = self.get_net(nx, ny)
        if cfg.use_mark:
            feed_dict = {
                net.inputs: iptdata,
                net.labels: gtdata,
                net.othermarks: get_mark_func((iptdata, gtdata),net.offsetx,(net.sx,net.sy)),
                net.keep_prob: cfg.keep_prob
            }
        else:
            feed_dict = {
                net.inputs: iptdata,
                net.labels: gtdata,
                net.keep_prob: cfg.keep_prob
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



    def _create_optimizer(self, optimizer = "Adam"):
        with tf.variable_scope('Model._optimizer'):
            if optimizer == "RMSProp":
                return tf.train.RMSPropOptimizer(learning_rate = args.learning_rate)
            elif optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            else:
                return None
 

if __name__ == '__main__':
    model = Model()