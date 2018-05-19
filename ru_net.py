import tensorflow as tf

from net.multilayer import create_ru_net_sp_init
from net.multilayer import calculate_offset
from net.cost import get_cost
from config import cfg

class Ru_net(object):
    def __init__(self,nx, ny, name):
        self.nx = nx
        self.ny = ny
        self.name = name
        self.channels = cfg.channels
        self.n_class = cfg.n_class
        self.sx, self.offsetx, self.sy, self.offsety = calculate_offset(nx, ny, cfg)
        
        if cfg.regularizer:
            regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.regularizer_scale)
        else:
            regularizer = None
        with tf.variable_scope('Ru_net'+name, reuse = tf.AUTO_REUSE,
                regularizer = regularizer):
            self.inputs = tf.placeholder(name = 'imgs', dtype = tf.float32, shape=[None, None, nx, ny, self.channels])
            self.labels = tf.placeholder(name = 'labels', dtype = tf.float32, shape=[None, None, nx, ny, self.n_class])
            self.othermarks = tf.placeholder(name = 'othermarks', dtype = tf.float32, shape=[None, None, self.sx, self.sy])
            self.keep_prob = tf.placeholder(name = 'keep_prob', dtype = tf.float32)
            self.firstframe = self.inputs[:,:1,:,:,:]
            self.otherframes = self.inputs[:,1:,:,:,:]
            self.firstlabel = self.labels[:,:1,:,:,:]
            self.cutlabels = self.labels[:,:,self.offsetx:self.offsetx + self.sx,self.offsety:self.offsety + self.sy,:]
            self.otherlabels = self.cutlabels[:,1:,:,:,:]
            self.predicts = create_ru_net_sp_init(self.nx, self.ny, self.firstframe, self.firstlabel, self.otherframes, self.channels, self.n_class, self.keep_prob, cfg)
            self.class_accuracy, self.total_accuracy, self.cost = get_cost(self.predicts, self.otherlabels, self.n_class, self.othermarks, regularizer, cfg)

if __name__ == '__main__':
    net = Ru_net(100, 100, 'testrunet')
    print(net.predicts)
    print(net.cost)
