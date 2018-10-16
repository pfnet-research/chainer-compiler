# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class Sum(chainer.Chain):
    def forward(self, x):
        return F.sum(x, axis=1)


class SumKeepdims(chainer.Chain):
    def forward(self, x):
        return F.sum(x, axis=0, keepdims=True)


class SumTupleAxis(chainer.Chain):
    def forward(self, x):
        return F.sum(x, axis=(1, 2))


class SumAllAxis(chainer.Chain):
    def forward(self, x):
        return F.sum(x)


# ======================================

import ch2o

if __name__ == '__main__':
    np.random.seed(314)
    a = np.random.rand(3, 5, 4).astype(np.float32)

    ch2o.generate_testcase(Sum(), [a])

    ch2o.generate_testcase(SumKeepdims(), [a], subname='keepdims')

    ch2o.generate_testcase(SumTupleAxis(), [a], subname='tuple_axis')

    ch2o.generate_testcase(SumAllAxis(), [a], subname='all_axis')
