# coding: utf-8

import numpy
import chainer
import chainer.links as L
import chainer.functions as F

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs,ps,p):
        t = [ xs[:i+1] for i in range(p) ]
        y1 =  [ v[-1] for v in t]
        #y1 = [xs for x in xs]
        y1 = F.pad_sequence(y1)
        return y1

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()
    
    wn = 1
    v = np.random.rand(10).astype(np.float32)
    w = np.random.randint(0,5,size=wn)
    p = np.int64(wn)
    chainer2onnx.generate_testcase(model, [v,w,p])
