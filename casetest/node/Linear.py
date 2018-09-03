# coding: utf-8

import chainer
import chainer.links as L

# Network definition


class A(chainer.Chain):

    def __init__(self, n_out):
        super(A, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None,n_out)

    def forward(self, x):
        return self.l1(x)


# ======================================

import testcasegen


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)
    
    model = A(3)

    x = np.random.rand(5,7).astype(np.float32)
    x = [x]
    testcasegen.generate_testcase(model, x)
