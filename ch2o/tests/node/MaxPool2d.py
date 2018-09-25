# coding: utf-8

import chainer
import chainer.functions as F


class A(chainer.Chain):
    def forward(self, x):
        y1 = F.max_pooling_2d(x, (1, 3), stride=(1, 4))
        return y1


class B(chainer.Chain):
    def forward(self, x):
        y1 = F.max_pooling_2d(x, (1, 3), stride=(1, 4), pad=(0, 1))
        return y1


# ======================================

import ch2o
import numpy as np

if __name__ == '__main__':
    x = np.random.rand(2, 3, 1, 13).astype(np.float32)
    ch2o.generate_testcase(A(), [x])
    ch2o.generate_testcase(B(), [x], subname='pad')
