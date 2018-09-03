# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs):
        return [ x[::-1] for x in xs]


# ======================================

import testcasegen


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(10,20).astype(np.float32)
    testcasegen.generate_testcase(model, [v])
