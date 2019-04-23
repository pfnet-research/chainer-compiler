# coding: utf-8

import chainer
import chainer.functions as F
import testtools

class ConcatTuple(chainer.Chain):
    def forward(self, x, y):
        return F.concat((x, y))


class ConcatList(chainer.Chain):
    def forward(self, x, y):
        return F.concat([x, y])


# ======================================

import numpy as np

def main():
    v = np.random.rand(7, 4, 2).astype(np.float32)
    w = np.random.rand(7, 3, 2).astype(np.float32)

    testtools.generate_testcase(ConcatTuple, [v, w])

    testtools.generate_testcase(ConcatList, [v, w], subname='list')

if __name__ == '__main__':
    main()
