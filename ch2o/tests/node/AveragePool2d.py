# coding: utf-8

import chainer
import chainer.functions as F


class AvgPool(chainer.Chain):
    def forward(self, x):
        y1 = F.average_pooling_2d(x, (1, 3), stride=(1, 4))
        return y1


class AvgPoolPad(chainer.Chain):
    def forward(self, x):
        y1 = F.average_pooling_2d(x, (3, 4), stride=1, pad=(2, 3))
        return y1


class AvgPoolNoStride(chainer.Chain):
    def forward(self, x):
        y1 = F.average_pooling_2d(x, (3, 4))
        return y1


# ======================================

import chainer_compiler.ch2o
import numpy as np

if __name__ == '__main__':

    x = np.random.rand(2, 3, 19, 13).astype(np.float32)
    ch2o.generate_testcase(AvgPool, [x])

    ch2o.generate_testcase(AvgPoolPad, [x], subname='pad')

    ch2o.generate_testcase(AvgPoolNoStride, [x], subname='no_stride')
