# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs,p):
        
        v = []
        for i in range(p):
            v.append(xs[:i])
        return v

# ======================================

import chainer2onnx


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(10).astype(np.float32)
    p = np.int64(5)
    chainer2onnx.generate_testcase(model, [v,p])
