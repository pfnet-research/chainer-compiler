# coding: utf-8

import chainer
import chainer.functions as F

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = F.tanh(x)
        return y1


# ======================================

import chainer2onnx 

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)
    
    model = A()

    x = np.random.rand(6,4).astype(np.float32)
    chainer2onnx.generate_testcase(model, [x])
