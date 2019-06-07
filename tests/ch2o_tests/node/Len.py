# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class Len(chainer.Chain):
    def forward(self, x):
        return len(x)


class LenList(chainer.Chain):
    def forward(self, xs):
        return len(xs)


# ======================================

from chainer_compiler import ch2o

if __name__ == '__main__':
    np.random.seed(314)

    ch2o.generate_testcase(Len(), [np.random.rand(3, 5, 4)])

    ch2o.generate_testcase(LenList(),
                           [[np.array(x) for x in [3, 5, 4]]],
                           subname='list')
