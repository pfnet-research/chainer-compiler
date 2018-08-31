# coding: utf-8

import chainer
import chainer.links as L


class BaN(chainer.Chain):

    def __init__(self):
        super(BaN, self).__init__()
        with self.init_scope():
            self.l1 = L.BatchNormalization(3)

    def forward(self, x):
        r = self.l1(x)
        print(r)
        return r


# ======================================from MLP

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    import test_mxnet

    model = BaN()

    v = np.random.rand(2, 3, 5, 5).astype(np.float32)
    print(v)
    test_mxnet.check_compatibility(model, v)
