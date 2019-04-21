# coding: utf-8

import chainer
import chainer.functions as F
import testtools
import numpy as np


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = F.swapaxes(x, 1, 3)
        y2 = F.swapaxes(x, 0, 1)
        return y1, y2


# ======================================


if __name__ == '__main__':

    model = A()

    x = np.random.rand(6, 4, 2, 7).astype(np.float32)
    testtools.generate_testcase(model, [x])
