# coding: utf-8

import chainer
import chainer.functions as F


class Matmul(chainer.Chain):
    def forward(self, x, y):
        return F.matmul(x, y)


# ======================================

import chainer_compiler.ch2o

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    x = np.random.rand(5, 7).astype(np.float32)
    y = np.random.rand(7, 4).astype(np.float32)
    ch2o.generate_testcase(Matmul, [x, y])
