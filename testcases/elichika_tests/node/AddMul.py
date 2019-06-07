# coding: utf-8

import chainer
from chainer_compiler.elichika import testtools
import numpy as np


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, y):
        z = (0.5 * 0.2) + 0.3 * x + y
        return z

def main():
    model = A()

    v = np.random.rand(3, 5).astype(np.float32)
    w = np.random.rand(3, 5).astype(np.float32)

    testtools.generate_testcase(model, [v, w])

if __name__ == '__main__':
    main()