# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools
import numpy as np


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = F.swapaxes(x, 1, 3)
        y2 = F.swapaxes(x, 0, 1)
        return y1, y2

class Self(chainer.Chain):

    def __init__(self):
        super(Self, self).__init__()

    def forward(self, x):
        y1 = x.swapaxes(1, 3)
        y2 = x.swapaxes(0, 1)
        return y1, y2

# ======================================


def main():
    model = A()

    x = np.random.rand(6, 4, 2, 7).astype(np.float32)
    testtools.generate_testcase(model, [x])

    testtools.generate_testcase(Self(), [x], subname='self')

if __name__ == '__main__':
    main()
