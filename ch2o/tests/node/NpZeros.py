# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class A(chainer.Chain):
    def forward(self):
        y1 = np.zeros((3, 4), dtype=np.float32)
        return y1


# ======================================

import ch2o

if __name__ == '__main__':
    ch2o.generate_testcase(A(), [])
