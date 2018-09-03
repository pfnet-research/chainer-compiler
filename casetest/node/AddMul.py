# coding: utf-8

import chainer
import chainer.functions as F


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, y):
        print(x, y)
        z = (0.5 * 0.2) + 0.3 * x + y
        return z


import numpy as np
import chainer2onnx
if __name__ == '__main__':
    model = A()

    v = np.random.rand(3, 5).astype(np.float32)
    w = np.random.rand(3, 5).astype(np.float32)

    chainer2onnx.generate_testcase(model, [v, w])
