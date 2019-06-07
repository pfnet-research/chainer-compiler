# coding: utf-8

import chainer
import chainer.functions as F


class A(chainer.Chain):
    def forward(self, x, y):
        y1 = F.hstack((x, y))
        return y1


class B(chainer.Chain):
    def forward(self, xs):
        y1 = F.hstack(xs)
        return y1


# ======================================

import chainer_compiler.ch2o
import numpy as np

if __name__ == '__main__':
    v = np.random.rand(5, 3, 2).astype(np.float32)
    w = np.random.rand(5, 4, 2).astype(np.float32)

    ch2o.generate_testcase(A(), [v, w])
    ch2o.generate_testcase(B(), [[v, w]], subname='list')
