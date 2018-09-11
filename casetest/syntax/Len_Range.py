# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs,p):
        # y1 = [np.int32(x) for x in range(p)]
        y1 =  [ xs[x,x+2] for x in range(p)]
        # y2 =  [ x[ps[0]:ps[0]+3] for x in xs]
        
        return y1

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(10,20).astype(np.float32)
    #w = np.random.randint(0,5,size=2)
    p = np.int64(3)
    chainer2onnx.generate_testcase(model, [v,p])
