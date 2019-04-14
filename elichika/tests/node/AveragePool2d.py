# coding: utf-8

import chainer
import chainer.functions as F


class AvgPool(chainer.Chain):

    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        y1 = F.average_pooling_2d(x, 1, stride=2)
        return y1


class AvgPoolPad(chainer.Chain):

    def __init__(self):
        super(AvgPoolPad, self).__init__()

    def forward(self, x):
        y1 = F.average_pooling_2d(x, 3, stride=1, pad=2)
        return y1


class AvgPoolNoStride(chainer.Chain):

    def __init__(self):
        super(AvgPoolNoStride, self).__init__()

    def forward(self, x):
        y1 = F.average_pooling_2d(x, 3)
        return y1

class AvgPoolPadTuple(chainer.Chain):
    def forward(self, x):
        y1 = F.average_pooling_2d(x, (3, 4), stride=1, pad=(2, 3))
        return y1

# ======================================

import testtools
import numpy as np


def main():
    np.random.seed(123)
    x = np.random.rand(2, 20, 15, 17).astype(np.float32)

    testtools.generate_testcase(AvgPool(), [x], subname='default')
    testtools.generate_testcase(AvgPoolPad(), [x], subname='withpad')
    testtools.generate_testcase(AvgPoolNoStride(), [x], subname='withoutstride')

    # TODO need to support it
    # testtools.generate_testcase(AvgPoolPadTuple(), [x], subname='withpadtuple')

if __name__ == '__main__':
    main()
