# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class B(chainer.Chain):
    def __init__(self, n_out, p):
        super(B, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, n_out)
        self.p = p

    def forward(self, x):
        x = self.l(x) * self.p
        return x


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(3)
            self.l1 = B(5, 3.1)
            self.l2 = B(4, 4.2)

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x) + self.l2.p
        x = self.l2(x) + self.l1.p
        return x


# ======================================

from chainer_compiler import ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    v = np.random.rand(3, 7).astype(np.float32)
    ch2o.generate_testcase(A, [v])
