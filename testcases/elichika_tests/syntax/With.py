# coding: utf-8

import chainer
from chainer_compiler.elichika.parser import flags

class X(object):
    def __init__(self, a):
        self.a = a

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False

class Y(X):
    def __enter__(self):
        self.a = self.a + 1
        return self

    def __exit__(self, type, value, traceback):
        self.a = self.a + 2
        return False

class Z(X):
    def __enter__(self):
        return self.a

# ======================================

class A(chainer.Chain):
    def forward(self):
        with X(3) as obj:
            return obj.a

class B(chainer.Chain):
    def forward(self):
        ret = 0
        with X(3) as obj1, X(4) as obj2:
            ret += obj1.a
        ret += obj2.a
        return ret

class C(chainer.Chain):
    def forward(self):
        ret = 0
        obj2 = Y(4)
        with Y(3) as obj1:
            ret += obj1.a
        ret +=  obj2.a
        return ret

class D(chainer.Chain):
    def forward(self):
        with Z(3) as obj:
            return obj

def func_elichika_dislike():
    yield 42

class IgnoreBranch(chainer.Chain):
    def forward(self):
        ret = 0
        with flags.ignore_branch():
            func_elichika_dislike()
        ret += 1
        return ret

class ForUnroll(chainer.Chain):
    def __init__(self):
        super(ForUnroll, self).__init__()
        self.x1 = 1
        self.x2 = 2
        self.x3 = 3
        self.dict = {'x1': 1, 'x2': 2, 'x3': 3}

    def forward(self):
        ret = 0

        with flags.for_unroll():
            for i in list([1, 2, 3]):
                ret += 1 * self['x%d' % i]

            for i in range(1, 4):
                ret += 2 * self['x%d' % i]

            for i in self.dict.keys():
                ret += 3 * self[i]

            for i in self.dict.values():
                ret += 4 * self['x%d' % i]

            with flags.for_unroll(unroll=False):
                for i in range(5):
                    ret += i

        return ret

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    testtools.generate_testcase(A(), [], 'basic')
    testtools.generate_testcase(B(), [], 'multiple')
    testtools.generate_testcase(C(), [], 'enter_exit')
    testtools.generate_testcase(D(), [], 'self_not_returned')
    testtools.generate_testcase(IgnoreBranch(), [], 'ignore_branch')
    testtools.generate_testcase(ForUnroll(), [], 'for_unroll')

if __name__ == '__main__':
    main()
