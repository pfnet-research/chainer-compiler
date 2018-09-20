# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, y):
        y1 = F.concat((x, y))
        # y2 = F.concat([y, x])
        return y1


# ======================================

import chainer2onnx

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(7, 4, 2).astype(np.float32)
    w = np.random.rand(7, 3, 2).astype(np.float32)

    chainer2onnx.generate_testcase(model, [v, w])
