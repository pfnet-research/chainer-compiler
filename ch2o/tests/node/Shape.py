# coding: utf-8

import chainer
import chainer.functions as F


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = x.shape
        return y1


# ======================================

import ch2o

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    x = np.random.rand(12, 6, 4).astype(np.float32)
    ch2o.generate_testcase(model, [x])
