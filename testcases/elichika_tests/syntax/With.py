# coding: utf-8

import chainer
from chainer_compiler.elichika.parser.flags import ignore_branch

class F(object):
    def __init__(self, a):
        self.a = a

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False

class A(chainer.Chain):
    def forward(self):
        with F(3) as obj:
            return obj.a

class B(chainer.Chain):
    def forward(self):
        ret = 0
        with F(3) as obj1, F(4) as obj2:
            ret = obj1.a + obj2.a
        return ret

class IgnoreBranch(chainer.Chain):
    def forward(self):
        ret = 0
        with ignore_branch():
            ret += 1
        ret += 1
        return ret


# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    testtools.generate_testcase(A(), [], 'basic')
    testtools.generate_testcase(B(), [], 'multiple')
    testtools.generate_testcase(IgnoreBranch(), [], 'ignore_branch')


if __name__ == '__main__':
    main()
