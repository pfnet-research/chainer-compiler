# coding: utf-8

import chainer
import numpy as np
from chainer_compiler.elichika import testtools

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        y1 = np.ceil(x)
        return y1


# ======================================

def main():
    model = A()

    x = (np.random.rand(6, 4).astype(np.float32) - 0.5) * 100.0
    testtools.generate_testcase(model, [x])

if __name__ == '__main__':
    main()