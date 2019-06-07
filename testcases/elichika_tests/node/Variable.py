# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import numpy as np
from chainer_compiler.elichika import testtools


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        return chainer.Variable(x)


class Self(chainer.Chain):
    def __init__(self, x):
        super(Self, self).__init__()
        self.x = chainer.Variable(x)
    def forward(self):
        return self.x

# ======================================

def main():
    testtools.generate_testcase(A(), [np.array(42)])

    testtools.generate_testcase(Self(np.array(42)), [], subname='self')

if __name__ == '__main__':
    main()