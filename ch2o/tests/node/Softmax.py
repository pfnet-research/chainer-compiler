# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class Softmax(chainer.Chain):
    def forward(self, x):
        return F.softmax(x)


class SoftmaxAxis(chainer.Chain):
    def forward(self, x):
        return F.softmax(x, axis=2)


# ======================================

from chainer_compiler import ch2o

if __name__ == '__main__':
    np.random.seed(314)
    a = np.random.rand(3, 5, 4).astype(np.float32)

    ch2o.generate_testcase(Softmax(), [a])

    ch2o.generate_testcase(SoftmaxAxis(), [a], subname='axis')
