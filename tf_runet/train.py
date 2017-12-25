from vot2016 import VOT2016_Data_Provider
from runet import RUNet
from util import oneHot_to_gray255

import numpy as np
from PIL import Image

def train(model_path = None, save_path = '/home/cjl/models/train/', max_step = 6, batch_size = 5, max_size = None, dataidx = 10):
    print('begin_train')
    data_provider = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    data_provider.dataidx = dataidx

    runet = RUNet('runet_train')
    if model_path is None:
        runet._init_vars_random()
    else:
        runet.restore(model_path)

    import psutil
    for i in range(10000):
        print('--------------------------------------')
        print('ite', i)
        iptdata, gtdata = data_provider.get_one_data_with_maxstep_next_batch(batch_size, max_step, max_size,runet.global_offset)
        #gtdataimg = oneHot_to_gray255(gtdata[0][0])
        #Image.fromarray(gtdataimg).show(title='0,5')
        #print('iptdata.shape:',iptdata.shape)
        #print('gtdata.shape:',gtdata.shape)
        cost, accuracy, otherlabels, predict = runet.train(iptdata, gtdata)
        print("cost:", cost, " accuracy:" , accuracy)
        otherlabels = otherlabels[0]
        predict = predict[0]
        '''
        sistep,six,siy,sic = otherlabels.shape
        slen = sistep * six * siy
        olrs = np.reshape(otherlabels, [slen,sic])
        zero = 0
        one = 0
        for il in range(slen):
            if olrs[il][0] == 1:
                zero = zero + 1
            else:
                one = one + 1
        print('zero:',zero)
        print('one',one)
        '''
        lbimg = oneHot_to_gray255(otherlabels[0])
        gtimg = oneHot_to_gray255(predict[0])
        #print('predict0:',predict[0])
        #print('otherlabels[0]:',otherlabels[0])
        img = np.append(lbimg,gtimg, axis=0)
        for step in range(1, 5 if 5 < max_step - 1 else max_step - 1):
            nimg = np.append(oneHot_to_gray255(otherlabels[step]),oneHot_to_gray255(predict[step]), axis=0)
            img = np.append(img, nimg, axis=1)
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        Image.fromarray(img).show(title='0,5')
        print('--------------------------------------')
        if (i % 10 == 0):
            filename = save_path + 'train' + str(i)
            runet.save(filename)
    print('========================================')

'''
def train_old(model_path = None):
    print('begin')
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    iptdata, gtdata = dptest.get_data_one_batch(8)
    iptdata = iptdata[:,0:10,:,:,:]
    gtdata = gtdata[:,0:10,:,:,:]

    runet = RUNet('runet_test')
    if model_path is None:
        runet._init_vars_random()
    else:
        runet.restore(model_path)

    import psutil
    for i in range(10000):
        print('--------------------------------------')
        print('ite', i)
        cost, accuracy, otherlabels, predict = runet.train(iptdata, gtdata)
        print("cost:", cost, " accuracy:" , accuracy)
        otherlabels = otherlabels[0]
        predict = predict[0]
        
        img = np.append(util.oneHot_to_gray255(otherlabels[0]),util.oneHot_to_gray255(predict[0]), axis=0)
        for step in range(1, 5):
            nimg = np.append(util.oneHot_to_gray255(otherlabels[step]),util.oneHot_to_gray255(predict[step]), axis=0)
            img = np.append(img, nimg, axis=1)
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        Image.fromarray(img).show(title='0,5')
        print('--------------------------------------')
        if (i % 50 == 0):
            filename = '/home/cjl/models/20171201/train' + str(i)
            runet.save(filename)
    print('========================================')
'''
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
        #model_path = '/home/cjl/models/20171202/train50',
        save_path = '/home/cjl/models/20171225/',
        max_step = 5,
        batch_size = 5,
        max_size = (300,300),
        dataidx = 9)