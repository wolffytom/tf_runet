from PIL import Image
import numpy as np
from util import oneHot_to_gray255

def display_softmax_2class_data(gtdata, ibatch, istep):
    batchsize, steps, nx, ny, nclass = np.shape(gtdata)
    assert(2 == nclass)
    assert(ibatch < batchsize and istep < steps)
    pic_np = oneHot_to_gray255(gtdata[ibatch][istep])
    return Image.fromarray(pic_np).show()
