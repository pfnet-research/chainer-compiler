# coding: utf-8

import chainer
import chainer.links as L


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.BatchNormalization(3)

    def forward(self, x):
        r = self.l1(x)
        return r


# ======================================from MLP

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(2, 3, 5, 5).astype(np.float32)

    import chainer_compiler.ch2o
    ch2o.generate_testcase(model, [v])
