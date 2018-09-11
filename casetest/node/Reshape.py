# coding: utf-8

import chainer
import chainer.functions as F

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, p):
        y1 = F.reshape(x,(24,12))
        y2 = F.reshape(x,(6,-1,12))
        y3 = F.reshape(x,(-1,p))
        return (y1,y2,y3)


# ======================================

import chainer2onnx 

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)
    
    model = A()

    x = np.random.rand(12,6,4).astype(np.float32)
    p = np.int64(3)
    chainer2onnx.generate_testcase(model, [x,p])
