# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, y):
        res = F.concat((x, y))
        return res


# ======================================

import testcasegen


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    # とりあえずというかんじ
    # これはなにかのtestになっているのだろうか

    model = A()

    v = np.random.rand(7, 4, 2).astype(np.float32)
    w = np.random.rand(7, 4, 2).astype(np.float32)

    testcasegen.generate_testcase(model, [v, w])
