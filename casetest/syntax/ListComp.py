# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs,ps,p):
        y1 =  [ x[3:5] for x in xs]
        y2 =  [ x[ps[0]:ps[0]+3] for x in xs]
        y3 = [ x[p:p+4] for x in xs]
        # y4 = [[ x[i:i+4] for x in xs] for i in range(p)]
        # y4 = [[ x for x in xs] for i in ps]
        # return y4
        return y1,y2,y3

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(10,20).astype(np.float32)
    w = np.random.randint(0,5,size=2)
    p = np.int64(3)
    ps = np.array([1,3,5])
    chainer2onnx.generate_testcase(model, [v,w,p])
