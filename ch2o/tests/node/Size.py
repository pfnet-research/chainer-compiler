# coding: utf-8

import chainer
import chainer.functions as F


class Size(chainer.Chain):
    def forward(self, x):
        y1 = x.size
        return y1


# ======================================

import ch2o

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    x = np.random.rand(12, 6, 4).astype(np.float32)

    ch2o.generate_testcase(Size(), [x])
