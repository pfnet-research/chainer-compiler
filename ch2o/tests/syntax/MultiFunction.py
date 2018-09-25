# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(7)
            self.l1 = L.Linear(5)

    def g(self, y):
        return self.l1(y)

    def forward(sl, x):
        x1 = sl.l0(x)
        x2 = sl.g(x1)
        return x2


# ======================================

import ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(10, 20).astype(np.float32)
    ch2o.generate_testcase(model, [v])
