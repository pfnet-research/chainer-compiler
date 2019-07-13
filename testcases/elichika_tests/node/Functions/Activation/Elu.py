# coding: utf-8

import chainer
import chainer.functions as F


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = F.elu(x)
        return y1

class Alpha(chainer.Chain):

    def __init__(self):
        super(Alpha, self).__init__()

    def forward(self, x):
        y1 = F.elu(x, alpha=0.9)
        return y1

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    x = np.random.rand(6, 4).astype(np.float32) - 0.5
    testtools.generate_testcase(A(), [x])

    testtools.generate_testcase(Alpha(), [x], subname='alpha')


if __name__ == '__main__':
    main()
