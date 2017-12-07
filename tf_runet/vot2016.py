import os
from PIL import Image
import numpy as np
import sys
from tqdm import * #pip3 install tqdm

class VOT2016_Data_Provider():
    def __init__(self,pathofvot2016):
        pathofinput = pathofvot2016 + '/input'
        pathofgroundtruth = pathofvot2016 + '/groundtruth'
        self.datanamelist = os.listdir(pathofgroundtruth)
        self.datanamesize = len(self.datanamelist)
        assert(self.datanamesize > 0)
        self.datalength = []
        self.inputdata = []
        self.gtdata = []
        self.maxsteps = -1
        self.minsteps = 99999999
        for idata in range(self.datanamesize):
            input_pic_dir = pathofinput + '/' + self.datanamelist[idata]
            input_gt_dir = pathofgroundtruth + '/' + self.datanamelist[idata]
            piclist = os.listdir(input_pic_dir)
            for inamen in range(len(piclist) - 1, -1,-1):
                iname = piclist[inamen]
                if (os.path.splitext(iname)[1] != '.jpg' and os.path.splitext(iname)[1] != '.png'):
                    piclist.remove(iname)
            piclist = sorted(piclist)
            gtlist = os.listdir(input_gt_dir)
            for inamen in range(len(gtlist) - 1, -1,-1):
                iname = gtlist[inamen]
                if (os.path.splitext(iname)[1] != '.jpg' and os.path.splitext(iname)[1] != '.png'):
                    gtlist.remove(iname)
            gtlist = sorted(gtlist)
            assert(len(gtlist) == len(piclist))
            datalength = len(gtlist)
            if datalength > self.maxsteps: self.maxsteps = datalength
            if datalength < self.minsteps: self.minsteps = datalength
            self.datalength.append(datalength)
            for inamen in range(datalength):
                piclist[inamen] = input_pic_dir + '/' + piclist[inamen]
                gtlist[inamen] = input_gt_dir + '/' + gtlist[inamen]
            self.inputdata.append(piclist)
            self.gtdata.append(gtlist)
        print(self.datalength)
        print(self.datanamelist)
        print('DataOK, loaded %d groups data.' % (len(self.datalength)))
        print('Max steps:%d' % (self.maxsteps))
        print('Min steps:%d' % (self.minsteps))
        self.dataidx = 0
        self.nowdata = None
        self.batchidx = 0
        #self.bagdata, self.baglabel = self.get_data(8)
    
    def get_data(self, dataidx):
        assert (0 <= dataidx and dataidx < self.datanamesize)
        datname = self.datanamelist[dataidx]
        inputnamelist = self.inputdata[dataidx]
        gtnamelist = self.gtdata[dataidx]
        steps = self.datalength[dataidx]
        im1 = Image.open(inputnamelist[0])
        im1_np = np.array(im1)
        nx = len(im1_np)
        ny = len(im1_np[0])
        channals = len(im1_np[0][0])
        assert(channals == 3)
        np.zeros(4)
        inputdata = np.zeros((steps, nx, ny, channals), dtype=np.float32)
        gtdata = np.zeros((steps, nx, ny), dtype = np.bool)
        print('loading data ' ,datname, '...')
        for istep in tqdm(range(steps)):
            im_ipt = Image.open(inputnamelist[istep])
            inputdata[istep] = np.array(im_ipt)
            im_gt = Image.open(gtnamelist[istep])
            gtdata[istep] = np.array(im_gt)
        gtdata = gtdata.reshape((steps*nx*ny))
        gtdata = gtdata.astype(np.int32)
        gtdataonehot = np.zeros((steps*nx*ny, 2), dtype=np.float32)
        gtdataonehot[np.arange(steps*nx*ny), gtdata] = 1
        gtdataonehot = gtdataonehot.reshape((steps,nx,ny,2))
        return (inputdata, gtdataonehot)

    def get_data_one_batch(self, dataidx):
        inputdata, gtdataonehot = self.get_data(dataidx)
        inputdata = inputdata.reshape([1] + list(np.shape(inputdata)))
        gtdataonehot = gtdataonehot.reshape([1] + list(np.shape(gtdataonehot)))
        # inputdata.dim:(batch_size = 1, steps, nx, ny, channals)
        # gtdata.dim:(batch_size = 1, steps, nx, ny, nclass)
        return (inputdata, gtdataonehot)

    def get_one_data_with_maxstep(self, dataidx, max_step):
        inputdata, gtdataonehot = self.get_data(dataidx)
        iptshp = list(np.shape(inputdata)) # iptdata is in shape[sheps, nx, ny, channels]
        gtshp = list(np.shape(gtdataonehot))
        steps = iptshp[0]
        if steps <= max_step:
            inputdata = inputdata.reshape([1] + list(np.shape(inputdata)))
            gtdataonehot = gtdataonehot.reshape([1] + list(np.shape(gtdataonehot)))
            return (inputdata, gtdataonehot)
        else:
            batch_size = steps // max_step
            inputdata = (inputdata[:batch_size * max_step,:,:,:]).reshape([batch_size, max_step, iptshp[1], iptshp[2],iptshp[3]])
            gtdataonehot = (gtdataonehot[:batch_size * max_step,:,:,:]).reshape([batch_size, max_step, gtshp[1], gtshp[2], gtshp[3]])
            return (inputdata, gtdataonehot)
    
    def subsampling(self, datatuple, max_nx, max_ny):
        inputdata, gtdataonehot = datatuple
        ##################
    
    def get_one_data_with_maxstep_next_batch(self, batch_size, max_step):
        if self.nowdata is None:
            self.nowdata = self.get_one_data_with_maxstep(self.dataidx, max_step)
        inputdata, gtdataonehot = self.nowdata
        batches = len(inputdata)
        if self.batchidx + batch_size >= batches:
            returndata = (inputdata[self.batchidx:batches], gtdataonehot[self.batchidx:batches])
        else:
            returndata = (inputdata[self.batchidx:self.batchidx + batch_size], gtdataonehot[self.batchidx:self.batchidx + batch_size])
        
        self.batchidx = self.batchidx + batch_size
        if self.batchidx >= batches:
            self.batchidx = 0
            self.dataidx = (self.dataidx + 1) % self.datanamesize
            self.nowdata = None
        
        return returndata

    def __call__(self, batch_size = 1):
        return self.bagdata, self.baglabel

def printlen():
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    for i in range (20):
        iptdata, gtdataonehot = dptest.get_data(i)
        nx = len(iptdata[0])
        ny = len(iptdata[0][0])
        print(nx, ' ',ny)

def test_maxstep():
    dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
    iptdata, gtdataonehot = dptest.get_one_data_with_maxstep_next_batch(10, 8)
    print(np.shape(iptdata))
    print(np.shape(gtdataonehot))
    #iptdata, gtdataonehot = dptest.get_one_data_with_maxstep_next_batch(10, 8)
    print(np.shape(iptdata))
    print(np.shape(gtdataonehot))
    #iptdata, gtdataonehot = dptest.get_one_data_with_maxstep_next_batch(10, 8)
    print(np.shape(iptdata))
    print(np.shape(gtdataonehot))


if __name__ == '__main__':
    #printlen()
    test_maxstep()
    