import os

class VOT2016_Data_Provider():
    def __init__(self,pathofvot2016):
        self.pathofinput = pathofvot2016 + '/input'
        self.pathofgroundtruth = pathofvot2016 + '/groundtruth'
        self.datanamelist = os.listdir(self.pathofgroundtruth)
        self.datanamesize = len(self.datanamelist)
        assert(self.datanamesize > 0)
        self.datalength = []
        self.inputdata = []
        self.gtdata = []
        self.maxsteps = -1
        self.minsteps = 99999999
        for idata in range(self.datanamesize):
            input_pic_dir = self.pathofinput + '/' + self.datanamelist[idata]
            input_gt_dir = self.pathofgroundtruth + '/' + self.datanamelist[idata]
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
            #print(piclist)
            #for i in range(len(piclist)):
            #    print(os.path.splitext(iname))
        print(self.datalength)
        print('DataOK, loaded %d groups data.' % (len(self.datalength)))
        print('Max steps:%d' % (self.maxsteps))
        print('Min steps:%d' % (self.minsteps))

dptest = VOT2016_Data_Provider('/home/cjl/data/vot2016')
