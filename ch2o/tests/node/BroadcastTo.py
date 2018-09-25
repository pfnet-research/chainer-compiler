# coding: utf-8

import chainer
import chainer.functions as F


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = F.broadcast_to(x, (2, 6, 4, 3))
        return y1


# ======================================

import ch2o
import numpy as np

if __name__ == '__main__':

    model = A()

    x = np.random.rand(6, 4, 1).astype(np.float32) - 0.5
    ch2o.generate_testcase(model, [x])
