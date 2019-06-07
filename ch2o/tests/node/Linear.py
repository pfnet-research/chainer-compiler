# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self, n_out):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_out)
            self.l2 = L.Linear(7, nobias=True)

    def forward(self, x):
        y1 = self.l1(x)
        y2 = self.l2(x)
        return (y1, y2)


class B(chainer.Chain):

    def __init__(self, n_out):
        super(B, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_out)

    def forward(self, x):
        return self.l1(x)


# ======================================

import chainer_compiler.ch2o

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    x = np.random.rand(5, 7).astype(np.float32)
    ch2o.generate_testcase(A(3), [x])

    x = np.random.rand(5, 7).astype(np.float32)
    ch2o.generate_testcase(B(3), [x], backprop=True)
