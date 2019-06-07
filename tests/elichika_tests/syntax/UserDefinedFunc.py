# coding: utf-8

import chainer
import tests.syntax.UserDefinedFuncSub1

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

class B(chainer.Chain):

    def __init__(self):
        super(B, self).__init__()

    def forward(self, x, y, z):
        return tests.syntax.UserDefinedFuncSub1.h(x, y)


class C(chainer.Chain):

    def __init__(self):
        super(C, self).__init__()

    def forward(self, x, y, z):
        p = tests.syntax.UserDefinedFuncSub1.F(x).g(y)
        return h(p, z)

# ======================================

import testtools
import numpy as np


def main():
    model = A()

    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(3, 4).astype(np.float32)
    testtools.generate_testcase(model, [a, b, c])
    testtools.generate_testcase(B(), [a, b, c], subname='external_func')
    testtools.generate_testcase(C(), [a, b, c], subname='external_class')

if __name__ == '__main__':
    main()
