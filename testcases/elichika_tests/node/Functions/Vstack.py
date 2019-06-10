# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools

class A(chainer.Chain):
    def forward(self, x, y):
        y1 = F.vstack((x, y))
        return y1


class B(chainer.Chain):
    def forward(self, xs):
        y1 = F.vstack(xs)
        return y1


# ======================================

import numpy as np

def main():
    v = np.random.rand(7, 4, 2).astype(np.float32)
    w = np.random.rand(5, 4, 2).astype(np.float32)

    testtools.generate_testcase(A(), [v, w])
    testtools.generate_testcase(B(), [[v, w]], subname='list')

if __name__ == '__main__':
    main()