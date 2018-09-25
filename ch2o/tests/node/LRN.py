# coding: utf-8

import chainer
import chainer.functions as F


class LRN(chainer.Chain):

    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        return F.local_response_normalization(x)


# ===========================================

import ch2o
import numpy as np

if __name__ == '__main__':

    model = LRN()
    v = np.random.rand(2, 3).astype(np.float32)
    ch2o.generate_testcase(model, [v])
