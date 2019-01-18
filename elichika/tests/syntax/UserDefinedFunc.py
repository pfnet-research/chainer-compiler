# coding: utf-8

import chainer


class F(object):
    def __init__(self, a):
        self.a = a

    def g(self, x):
        return self.a + x


def h(x, y):
    return x + y


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, y, z):
        p = F(x).g(y)
        return h(p, z)


# ======================================

import testtools
import numpy as np

if __name__ == '__main__':
    model = A()

    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(3, 4).astype(np.float32)
    testtools.generate_testcase(model, [a, b, c])
