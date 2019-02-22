# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class Array(chainer.Chain):
    def forward(self):
        y1 = np.array([4.0, 2.0, 3.0], dtype=np.float32)
        return y1


class ArrayCast(chainer.Chain):
    def forward(self):
        y1 = np.array([4.0, 2.0, 3.0], dtype=np.int32)
        return y1


# ======================================

import testtools
import numpy as np

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    testtools.generate_testcase(Array, [])
    testtools.generate_testcase(ArrayCast, [], subname='cast')
