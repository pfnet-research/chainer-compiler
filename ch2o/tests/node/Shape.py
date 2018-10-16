# coding: utf-8

import chainer
import chainer.functions as F


class Shape(chainer.Chain):
    def forward(self, x):
        y1 = x.shape
        return y1


class ShapeConcat(chainer.Chain):
    def forward(self, x):
        y1 = x.shape
        return np.array(y1 + (42,))


# ======================================

import ch2o

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    x = np.random.rand(12, 6, 4).astype(np.float32)

    ch2o.generate_testcase(Shape(), [x])

    ch2o.generate_testcase(ShapeConcat(), [x])
