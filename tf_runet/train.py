from vot2016 import VOT2016_Data_Provider
from runet import RUNet
from util import oneHot_to_gray255

import tensorflow as tf
import numpy as np
from PIL import Image

def train(model_path = None,
          save_path = '/home/cjl/models/train',
          data_path = '/home/cjl/data/vot2016',
          max_step = 6,
          batch_size = 5,
          max_size = None,
          total_step = 0,
          display = False,
          displaystep = 30,
          dataidx = 10):
    print('begin_train')
    data_provider = VOT2016_Data_Provider(data_path)
    data_provider.dataidx = dataidx

    runet = RUNet('runet_train')
    train_writer = tf.summary.FileWriter(save_path, runet.sess.graph)
    if model_path is None:
        runet._init_vars_random()
    else:
        runet.restore(model_path)

    import psutil
    training = True
    while training:
        total_step += 1
        print('--------------------------------------')
        print('total_step:', total_step)
        iptdata, gtdata = data_provider.get_one_data_with_maxstep_next_batch(batch_size, max_step, max_size,runet.global_offset)
        summary, cost, total_accuracy, class_accuracy, otherlabels, predict = runet.train(iptdata, gtdata, data_provider.get_mark)
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
        if (total_step % 10 == 0):
            filename = save_path + '/train' + str(total_step)
            runet.save(filename)
    print('========================================')

def predict(model_path):
    print('begin')
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    iptdata, gtdata = dptest.get_data_one_batch(8)
    iptdata = iptdata[:,0:10,:,:,:]
    gtdata = gtdata[:,0:10,:,:,:]

    runet = RUNet('runet_test')
    runet.restore(model_path)

    cost, accuracy, otherlabels, predict = runet.predict(iptdata, gtdata)
    print("cost:", cost, " accuracy:" , accuracy)
    otherlabels = otherlabels[0]
    predict = predict[0]
        
    img = np.append(util.oneHot_to_gray255(otherlabels[0]),util.oneHot_to_gray255(predict[0]), axis=0)
    for step in range(1, 5):
        nimg = np.append(util.oneHot_to_gray255(otherlabels[step]),util.oneHot_to_gray255(predict[step]), axis=0)
        img = np.append(img, nimg, axis=1)
    Image.fromarray(img).show(title='0,5')

if __name__ == '__main__':
    #train()
    #train('/home/cjl/models/20171127/train200')
    #newclass()
    #predict('/home/cjl/models/20171201/train150')
    train(
        #model_path = '/home/cjl/models/20180102/train100',
        save_path = '/home/cjl/models/20180104',
        max_step = 10,
        batch_size = 8,
        max_size = (300,300),
        dataidx = 10)