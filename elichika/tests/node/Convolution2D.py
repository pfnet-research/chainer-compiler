# coding: utf-8

import chainer
import chainer.links as L
from chainer import serializers

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
        with self.init_scope():
            # TODO Add more tests
            self.l1 = L.Convolution2D(None, 6, (5, 7), stride=(2, 3))

    def forward(self, x):
        y1 = self.l1(x)
        return y1


# ======================================

import testtools
import numpy as np


def main():
    model = A()

    np.random.seed(123)
    x = np.random.rand(2, 20, 15, 17).astype(np.float32)

    model(x)

    testtools.generate_testcase(model, [x])


if __name__ == '__main__':
    main()
