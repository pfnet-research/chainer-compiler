# coding: utf-8

import chainer
import chainer.functions as F


class Default(chainer.Chain):

    def __init__(self):
        super(Default, self).__init__()

    def forward(self, x):
        y1 = F.leaky_relu(x)
        return y1

class Slope(chainer.Chain):

    def __init__(self):
        super(Slope, self).__init__()

    def forward(self, x):
        y1 = F.leaky_relu(x, slope=0.1)
        return y1

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    x = np.random.rand(6, 4).astype(np.float32) - 0.5
    testtools.generate_testcase(Default(), [x])

    testtools.generate_testcase(Slope(), [x], subname='slope')


if __name__ == '__main__':
    main()
