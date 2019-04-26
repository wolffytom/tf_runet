#!/usr/bin/env python3


import os, sys
sys.path.append( os.path.dirname(__file__ ) )
from data.vot2016 import VOT2016_Data_Provider
from config import cfg
from model import Model
import sklearn

import tensorflow as tf
import numpy as np
from PIL import Image
from meval import calc_auc, print_step_auc
import text_histogram

def ave_weight(gtdata):
    gtdata_flat = gtdata.reshape((-1))
    weight = np.zeros(shape=gtdata_flat.shape)
    ones = np.sum(gtdata)
    length = len(gtdata_flat)
    zeros = length-ones
    zeros_weight = ones/zeros
    for i in range(len(gtdata_flat)):
        if gtdata_flat[i] > 0.5:
            weight[i] = 1.
        else:
            weight[i] = zeros_weight
    weight = weight.reshape(gtdata.shape)
    return weight

def train(model_path = None,
          save_path = '/home/cjl/tf_runet/models/20180612',
          pro_path = '/home/cjl/tf_runet',
          max_size = None,
          total_step = 0,
          display = False,
          displaystep = 30,
          save = True,
          dataidx = 10):
    print('begin_train')
    data_path = pro_path + '/data/vot2016'
    data_provider = VOT2016_Data_Provider(data_path, cfg)
    data_provider.random_batch_init()
    data_provider.dataidx = dataidx

    model = Model()
    train_writer = tf.summary.FileWriter(save_path, model.sess.graph)
    if model_path is None:
        model.init_vars_random()
    else:
        model.restore(model_path)

    import psutil
    training = True
    while training:
        total_step += 1
        print('--------------------------------------')
        print('total_step:', total_step)
        iptdata, gtdata = data_provider.get_a_random_batch(jump=2)
        weight = ave_weight(gtdata)
        summary, cost, otherlabels, predict = model.train(iptdata, gtdata, weight)
        #text_histogram.histogram(list(iptdata.astype(float).reshape((-1))))
        #text_histogram.histogram(list(gtdata.astype(float).reshape((-1))))
        auc = calc_auc(predict, otherlabels)
        print("cost:", cost, " auc:" , auc)
        print_step_auc(predict, otherlabels)
        text_histogram.histogram(list(predict.astype(float).reshape((-1))))

        train_writer.add_summary(summary, total_step)
        if (save and total_step % 20 == 0):
            filename = save_path + '/train' + str(total_step)
            model.save(filename)
    print('========================================')

if __name__ == '__main__':
    #train()
    #train('/home/cjl/models/20171127/train200')
    #newclass()
    #predict('/home/cjl/models/20171201/train150')

    scripts_path = os.path.split( os.path.realpath( sys.argv[0] ) )[0]
    train(
        pro_path = scripts_path,
        model_path = None,
        save_path = scripts_path + '/models/0402_all',
        max_size = (300,300),
        dataidx = 10)
