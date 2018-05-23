from data.vot2016 import VOT2016_Data_Provider
from config import cfg
from model import Model

import tensorflow as tf
import numpy as np
from PIL import Image

import os
import sys

def train(model_path = None,
          save_path = '/home/cjl/tf_runet/models/20180523',
          pro_path = '/home/cjl/tf_runet',
          max_size = None,
          total_step = 0,
          display = False,
          displaystep = 30,
          save = False,
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
        #iptdata, gtdata = data_provider.get_one_data_with_maxstep_next_batch(cfg.batch_size, cfg.max_step, max_size,model.offset)
        iptdata, gtdata = data_provider.get_a_random_batch()
        summary, cost, total_accuracy, class_accuracy, otherlabels, predict = model.train(iptdata, gtdata, data_provider.get_mark)
        print("cost:", cost, " total_accuracy:" , total_accuracy, " class_accuracy:" , class_accuracy)
        train_writer.add_summary(summary, total_step)
        if display and (total_step%displaystep) == 0:
            otherlabels = otherlabels[0]
            predict = predict[0]
            lbimg = oneHot_to_gray255(otherlabels[0])
            gtimg = oneHot_to_gray255(predict[0])
            img = np.append(lbimg,gtimg, axis=0)
            for step in range(1, 9 if 9 < max_step - 1 else max_step - 1):
                nimg = np.append(oneHot_to_gray255(otherlabels[step]),oneHot_to_gray255(predict[step]), axis=0)
                img = np.append(img, nimg, axis=1)
            for proc in psutil.process_iter():
                if proc.name() == "display":
                    proc.kill()
            Image.fromarray(img).show(title='0,5')
        print('--------------------------------------')
        if (save and total_step % 10 == 0):
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
        save_path = scripts_path + '/models/20180522',
        max_size = (300,300),
        dataidx = 10)
