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

class IgnoreBranch(chainer.Chain):
    def forward(self):
        ret = 0
        with flags.ignore_branch():
            ret += 1
        ret += 1
        return ret


# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    testtools.generate_testcase(A(), [], 'basic')
    testtools.generate_testcase(B(), [], 'multiple')
    testtools.generate_testcase(C(), [], 'enter_exit')
    testtools.generate_testcase(D(), [], 'self_not_returned')
    # testtools.generate_testcase(IgnoreBranch(), [], 'ignore_branch')


if __name__ == '__main__':
    main()
