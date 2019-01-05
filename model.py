#
import numpy as np
import tensorflow as tf

from ru_net import Ru_net
from config import cfg
from sess import get_sess

class Model(object):
    def __init__(self):
        self._nets = {}
        self._base_net = self.get_net(cfg.base_net_size, cfg.base_net_size)
        self._optimizer = self._create_optimizer(cfg.optimizer)
        self._base_net_minimizer = self._optimizer.minimize(self._base_net.cost)
        if cfg.useGPU:
            self.sess = tf.Session()#get_sess()
        else:
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
        self.offset = self._base_net.offsetx

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

    def init_vars_random(self):
        self.sess.run(tf.global_variables_initializer())

    def save(self, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, model_path)
        print("Model saved in file: %s" % save_path)
        return save_path

    def restore(self, model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        print("Model restored from file: %s" % model_path)

    def get_net(self, nx, ny):
        scrpt = False
        netname = str(nx) + 'x' + str(ny)
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

    def train(self, iptdata, gtdata, get_mark_func,
            print_datainfo = False):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert cfg.channels == channels
        n_class = np.shape(gtdata)[4]
        assert cfg.n_class == n_class

        if print_datainfo:
            print('iptdata.shape:', iptdata.shape)
            print('gtdata.shape:', gtdata.shape)

        net = self.get_net(nx, ny)
        if cfg.use_mark:
            othermarks = get_mark_func((iptdata, gtdata),net.offsetx,(net.sx,net.sy))
            feed_dict = {
                net.inputs: iptdata,
                net.labels: gtdata,
                net.othermarks: othermarks,
                net.keep_prob: cfg.keep_prob
            }
        else:
            feed_dict = {
                net.inputs: iptdata,
                net.labels: gtdata,
                net.keep_prob: cfg.keep_prob
            }

        _opt, cost, total_accuracy, class_accuracy, otherlabels, predict = self.sess.run((
            self._optimizer.minimize(net.cost),
            net.cost,
            net.total_accuracy,
            net.class_accuracy,
            net.otherlabels,
            net.predicts), feed_dict=feed_dict)

        summary = self.sess.run(tf.summary.merge_all(), feed_dict={
            self.cost:cost,
            self.total_accuracy:total_accuracy,
            self.class_accuracy:class_accuracy,
            self.predict:predict})
        return summary, cost, total_accuracy, class_accuracy, otherlabels, predict

    def _create_optimizer(self, optimizer = "Adam"):
        with tf.variable_scope('Model._optimizer'):
            if optimizer == "RMSProp":
                return tf.train.RMSPropOptimizer(learning_rate = cfg.learning_rate)
            elif optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=cfg.learning_rate)
            else:
                return None


if __name__ == '__main__':
    model = Model()
