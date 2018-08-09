# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L

# Network definition


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# ======================================from MLP

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    import test_mxnet

    model = MLP(8, 2)

    v = np.random.rand(5, 3).astype(np.float32)
    test_mxnet.check_compatibility(model, v)
