import numpy as np
import tensorflow as tf

from ru_net import Ru_net
from config import cfg
from sess import get_sess

class Model(object):
    def __init__(self):
        self._nets = {}
        self._base_net = self.get_net(cfg['base_net_size'], cfg['base_net_size'])
        self._optimizer = self._create_optimizer(cfg['optimizer'])
        self._base_net_minimizer = self._optimizer.minimize(self._base_net.cost)
        if cfg['useGPU']:
            self.sess = tf.Session()#get_sess()
        else:
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
        self.offset = self._base_net.offsetx

        self.cost = tf.placeholder(name = 'runet.cost', dtype = tf.float32, shape=None)
        tf.summary.scalar('cost', self.cost)
        self.predict = tf.placeholder(dtype = tf.float32, shape=[None, None, None, None])
        predict_flat = tf.reshape(self.predict, [-1])

    def init_vars_random(self):
        self.sess.run(tf.local_variables_initializer())
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

    def train(self, iptdata, gtdata, weight,
            print_datainfo=False, eval_=False):
        iptdata_shape = np.shape(iptdata)
        batch_size, steps, nx, ny, channels = iptdata_shape
        assert cfg['channels'] == channels

        if print_datainfo:
            print('iptdata.shape:', iptdata.shape)
            print('gtdata.shape:', gtdata.shape)

        net = self.get_net(nx, ny)
        if True:
            feed_dict = {
                net.inputs: iptdata,
                net.labels: gtdata,
                net.weights: weight,
                net.keep_prob: cfg['keep_prob']
            }

        cost, otherlabels, predict = None, None, None
        if eval_:
            cost, otherlabels, predict = self.sess.run((
                net.cost,
                net.otherlabels,
                net.predicts), feed_dict=feed_dict)
        else:
            _, cost, otherlabels, predict = self.sess.run((
                self._optimizer.minimize(net.cost),
                net.cost,
                net.otherlabels,
                net.predicts), feed_dict=feed_dict)

        summary = self.sess.run(tf.summary.merge_all(), feed_dict={
            self.cost:cost, 
            self.predict:predict})
        return summary, cost, otherlabels, predict

    def _create_optimizer(self, optimizer = "Adam"):
        with tf.variable_scope('Model._optimizer'):
            if optimizer == "RMSProp":
                return tf.train.RMSPropOptimizer(learning_rate = cfg['learning_rate'])
            elif optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=cfg['learning_rate'])
            else:
                return None
 

if __name__ == '__main__':
    model = Model()
