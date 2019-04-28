# coding: utf-8

import chainer
import chainer.functions as F


class Shape(chainer.Chain):
    def forward(self, x):
        y1 = x.shape
        return list(y1)


class ShapeConcat(chainer.Chain):
    def forward(self, x):
        y1 = x.shape
        return np.array(y1 + (42,))


# ======================================

import testtools
import numpy as np

def main():

    x = np.random.rand(12, 6, 4).astype(np.float32)

    testtools.generate_testcase(Shape(), [x])
    testtools.generate_testcase(ShapeConcat(), [x], subname='concat')


if __name__ == '__main__':
    main()
