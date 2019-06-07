# coding: utf-8

import chainer
import numpy as np


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = np.ceil(x)
        return y1


# ======================================

import chainer_compiler.ch2o

if __name__ == '__main__':

    model = A()

    x = (np.random.rand(6, 4).astype(np.float32) - 0.5) * 100.0
    ch2o.generate_testcase(model, [x])
