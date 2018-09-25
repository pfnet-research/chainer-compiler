# coding: utf-8

import chainer

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        return x

# ======================================


import ch2o

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    x = np.random.rand(5, 7).astype(np.float32)
    x = [x]
    ch2o.generate_testcase(model, x)
