# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        return chainer.Variable(x)


# ======================================

import chainer_compiler.ch2o

if __name__ == '__main__':
    ch2o.generate_testcase(A(), [np.array(42)])
