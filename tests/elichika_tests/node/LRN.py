# coding: utf-8

import chainer
import chainer.functions as F


class LRN(chainer.Chain):

    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        return F.local_response_normalization(x)


# ===========================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    np.random.seed(314)
    v = np.random.rand(2, 3).astype(np.float32)

    testtools.generate_testcase(LRN, [v], subname='basic')

if __name__ == '__main__':
    main()
