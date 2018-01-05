from runet import RUNet
import tensorflow as tf

def drawgraph(save_path):
    runet = RUNet('Show_Graph')
    train_writer = tf.summary.FileWriter(save_path, runet.sess.graph)
    runet.sess.run(runet.cost, feed_dict={runet.cost:1.0})

if __name__ == '__main__':
    drawgraph(save_path = '/home/cjl/models/20180105')