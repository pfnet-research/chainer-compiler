# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs,ps,p):
        y1 =  [ xs[x,x+2] for x in range(p)]
        y2 =  [ xs[ps[x],ps[x]+3] for x in range(p)]
        return y1,y2

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()
    
    wn = 5
    v = np.random.rand(10,20).astype(np.float32)
    w = np.random.randint(0,5,size=wn)
    p = np.int64(wn)
    chainer2onnx.generate_testcase(model, [v,w,p])
