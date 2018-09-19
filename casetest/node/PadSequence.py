# coding: utf-8

import chainer
import chainer.functions as F

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs):
        y1 = F.pad_sequence(xs)
        return y1


# ======================================

import chainer2onnx 

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)
    
    model = A()
    
    ls = np.random.randint(0,10,size=5)
    x = [np.random.rand(i).astype(np.float32) for i in ls]
    chainer2onnx.generate_testcase(model, [x])
