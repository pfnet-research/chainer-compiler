# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class Mean(chainer.Chain):
    def forward(self, x):
        return F.mean(x, axis=1)


class MeanKeepdims(chainer.Chain):
    def forward(self, x):
        return F.mean(x, axis=0, keepdims=True)


class MeanTupleAxis(chainer.Chain):
    def forward(self, x):
        return F.mean(x, axis=(1, 2))


class MeanAllAxis(chainer.Chain):
    def forward(self, x):
        return F.mean(x)


# ======================================

from chainer_compiler import ch2o

if __name__ == '__main__':
    np.random.seed(314)
    a = np.random.rand(3, 5, 4).astype(np.float32)

    ch2o.generate_testcase(Mean(), [a])

    ch2o.generate_testcase(MeanKeepdims(), [a], subname='keepdims')

    ch2o.generate_testcase(MeanTupleAxis(), [a], subname='tuple_axis')

    ch2o.generate_testcase(MeanAllAxis(), [a], subname='all_axis')
