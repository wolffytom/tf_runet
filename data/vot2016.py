import os
from PIL import Image
import numpy as np
import random
import sys
#from tqdm import * #pip3 install tqdm

class VOT2016_Data_Provider():
    def __init__(self,pathofvot2016, cfg):
        self.cfg = cfg
        pathofinput = pathofvot2016 + '/images'
        pathofgroundtruth = pathofvot2016 + '/groundtruths'
        self.datanamelist = os.listdir(pathofgroundtruth)
        self.datanamesize = len(self.datanamelist)
        assert(self.datanamesize > 0)
        self.datalength = []
        self.gtedge = []
        self.inputdata = []
        self.gtdata = []
        self.nxs = []
        self.nys = []
        self.maxsteps = -1
        self.minsteps = 99999999
        self.channals = cfg['channels']
        self.n_class = 2
        self.batch_size = cfg['batch_size']
        self.steps = cfg['max_step']
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
            self.gtedge.append(None)
            im1 = Image.open(piclist[0])
            im1_np = np.array(im1)
            self.nxs.append(len(im1_np))
            self.nys.append(len(im1_np[0]))
            assert(len(im1_np[0][0]) == self.channals)
        self.dataidx = 0
        self.nowdata = None
        self.batchidx = 0
        #self.bagdata, self.baglabel = self.get_data(8)

    def random_batch_init(self):
        self.batches = []
        for dataidx in range(self.datanamesize):
            start = 0
            while (True):
                end = start + self.batch_size*self.steps - 1
                if end >= self.datalength[dataidx]:
                    break
                abatch = (dataidx, start)
                self.batches.append(abatch)
                start = end + 1
        self.batch_nums = len(self.batches)
        print(str(self.batch_nums) + ' batches ready')

    def get_images(self, dataidx, start, steps, jump=1):
        nx, ny = 256,256
        inputdata = np.zeros((steps, nx, ny, self.channals), dtype=np.float32)
        gtdata = np.zeros((steps, nx, ny), dtype = np.bool)
        inputnamelist = self.inputdata[dataidx]
        gtnamelist = self.gtdata[dataidx]
        for istep in range(steps):#tqdm(range(steps)):
            im_ipt = Image.open(inputnamelist[start + istep*jump])
            im_ipt = im_ipt.resize((nx, ny))
            inputdata[istep] = np.array(im_ipt)
            im_gt = Image.open(gtnamelist[start + istep*jump])
            im_gt = im_gt.resize((nx, ny))
            gtdata[istep] = np.array(im_gt)
        if self.cfg['norm_input']:
            if self.cfg['norm_input_minus']:
                inputdata = (inputdata * 2 - 255) / 255
            else:
                inputdata = (inputdata / 255.0)
        gtdata = gtdata.astype(np.float)
        return (inputdata, gtdata)

    def get_a_random_batch(self, jump=1):
        #batchidx = random.randint(0, self.batch_nums-1)
        batchidx = 1
        dataidx, start = self.batches[batchidx]
        inputdata, gtdataonehot = self.get_images(dataidx, start, self.batch_size * self.steps, jump=jump)
        nsteps, nx, ny, band = inputdata.shape
        inputdata = inputdata.reshape((self.batch_size, self.steps, nx, ny, self.channals))
        gtdata = gtdataonehot.reshape((self.batch_size, self.steps, nx, ny))
        datatuple = (inputdata, gtdata)
        if self.cfg['use_max_size']:
            datatuple = self.subsampling(datatuple, (self.cfg['max_size_x'], self.cfg['max_size_y']))
        return datatuple 

    def get_data(self, dataidx):
        assert (0 <= dataidx and dataidx < self.datanamesize)
        datname = self.datanamelist[dataidx]
        inputnamelist = self.inputdata[dataidx]
        gtnamelist = self.gtdata[dataidx]
        steps = self.datalength[dataidx]
        nx = self.nxs[dataidx]
        ny = self.nys[dataidx]
        channals = self.channals
        inputdata = np.zeros((steps, nx, ny, channals), dtype=np.float32)
        gtdata = np.zeros((steps, nx, ny), dtype = np.bool)
        print('loading data ' ,datname, '...')
        for istep in range(steps):#tqdm(range(steps)):
            im_ipt = Image.open(inputnamelist[istep])
            inputdata[istep] = np.array(im_ipt)
            im_gt = Image.open(gtnamelist[istep])
            gtdata[istep] = np.array(im_gt)
        gtdata = gtdata.astype(np.int32)
        gtdata = gtdata.reshape((steps*nx*ny))
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

    def get_one_data_with_maxstep(self, max_step):
        inputdata, gtdataonehot = self.get_data(self.dataidx)
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

    def subsampling(self, datatuple, max_size):
        max_nx, max_ny = max_size
        inputdata, gtdata = datatuple
        batch_size,steps,nx,ny,channels = inputdata.shape
        timex = (nx + max_nx - 1) // max_nx
        timey = (ny + max_ny - 1) // max_ny
        time = timex if timex > timey else timey
        return (inputdata[:,:,::time,::time,:], gtdata[:,:,::time,::time])
    
    def get_one_data_with_maxstep_next_batch_t(self, batch_size, max_step, max_size = None, edge = 0):
        if self.nowdata is None:
            self.nowdata = self.get_one_data_with_maxstep(max_step)
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
        
        if max_size is not None:
            returndata = self.subsampling(returndata, max_size)
        
        gtdata = returndata[1]
        _,__,nx,ny,___ = gtdata.shape
        sumcenter = np.sum(gtdata[:,:,edge:nx-edge,edge:ny-edge,1:2])
        sumedge = np.sum(gtdata[:,:,:,:,1:2]) - sumcenter
        #print(gtdata)
        if sumcenter > 0 and sumedge == 0:
            return returndata
        else:
            #print('sumcenter:',sumcenter)
            #print('sumedge:',sumedge)
            #return self.get_one_data_with_maxstep_next_batch(batch_size, max_step, max_size, edge)
            return None

    def get_one_data_with_maxstep_next_batch(self, batch_size, max_step, max_size = None, edge = 0, centershape = None):
        rd = self.get_one_data_with_maxstep_next_batch_t(batch_size,max_step,max_size,edge)
        while rd is None:
            rd = self.get_one_data_with_maxstep_next_batch_t(batch_size,max_step,max_size,edge)
        return rd[0], rd[1]#, self.get_mark(rd, edge, centershape)

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
    dptest.dataidx = 9
    iptdata, gtdataonehot = dptest.get_one_data_with_maxstep_next_batch(10, 8, max_size = (30,30))
    print(np.shape(iptdata))
    print(np.shape(gtdataonehot))
    #iptdata, gtdataonehot = dptest.get_one_data_with_maxstep_next_batch(10, 8)
    print(np.shape(iptdata))
    print(np.shape(gtdataonehot))
    #iptdata, gtdataonehot = dptest.get_one_data_with_maxstep_next_batch(10, 8)
    print(np.shape(iptdata))
    print(np.shape(gtdataonehot))

def test_resize():
    import sys
    sys.path.append('/home/cjl/tf_runet')
    from config import cfg
    pro_path = '/home/cjl/tf_runet'
    data_path = pro_path + '/data/vot2016'
    data_provider = VOT2016_Data_Provider(data_path, cfg)
    data_provider.random_batch_init()
    data_provider.dataidx = 10
    iptdata, gtdata = data_provider.get_a_random_batch(jump=20)

if __name__ == '__main__':
    #printlen()
    #test_maxstep()
    test_resize()
    
