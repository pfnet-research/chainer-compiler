# coding: utf-8

import chainer


class Len(chainer.Chain):

    def __init__(self):
        super(Len, self).__init__()

    def forward(self, x):
        return len(x)


class LenList(chainer.Chain):

    def __init__(self):
        super(LenList, self).__init__()

    def forward(self, xs):
        return len(xs)


# ======================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    np.random.seed(314)

    testtools.generate_testcase(Len(), [np.random.rand(3, 5, 4)], subname='basic')
    testtools.generate_testcase(LenList(), [[np.array(x) for x in [3, 5, 4]]], subname='list')


if __name__ == '__main__':
    main()
