# coding: utf-8

import chainer
import chainer.functions as F


class Unpooling2D(chainer.Chain):
    def forward(self, x):
        y = F.unpooling_2d(x, 2, cover_all=False)
        return y


class Unpooling2D_3x4(chainer.Chain):
    def forward(self, x):
        y = F.unpooling_2d(x, (3, 4), cover_all=False)
        return y


# ======================================

import numpy as np
import testtools

def main():
    x = np.random.rand(2, 3, 11, 7).astype(np.float32)
    testtools.generate_testcase(Unpooling2D, [x])
    testtools.generate_testcase(Unpooling2D_3x4, [x], subname='3x4')

    # The largest input in FPN.
    x = np.random.rand(1, 256, 100, 100).astype(np.float32)
    testtools.generate_testcase(Unpooling2D, [x], subname='large')

if __name__ == '__main__':
    main()
