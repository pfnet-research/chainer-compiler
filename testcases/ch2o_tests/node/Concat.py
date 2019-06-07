# coding: utf-8

import chainer
import chainer.functions as F


class ConcatTuple(chainer.Chain):
    def forward(self, x, y):
        return F.concat((x, y))


class ConcatList(chainer.Chain):
    def forward(self, x, y):
        return F.concat([x, y])


# ======================================

from chainer_compiler import ch2o
import numpy as np

if __name__ == '__main__':
    v = np.random.rand(7, 4, 2).astype(np.float32)
    w = np.random.rand(7, 3, 2).astype(np.float32)

    ch2o.generate_testcase(ConcatTuple, [v, w])

    ch2o.generate_testcase(ConcatList, [v, w], subname='list')
