# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = F.dropout(x)
        return y1


# ======================================

import numpy as np

def main():

    model = A()

    x = np.random.rand(6, 4).astype(np.float32)
    testtools.generate_testcase(model, [x])

if __name__ == '__main__':
    main()
