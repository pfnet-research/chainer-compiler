# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        loss = F.softmax_cross_entropy(h3, t)
        return loss


from chainer_compiler.elichika import testtools
import numpy as np

def main():
    np.random.seed(314)

    out_n = 4
    batch_size = 100
    model = MLP(8, out_n)

    v = np.random.rand(batch_size, 3).astype(np.float32)
    w = np.random.randint(out_n, size=batch_size)
    testtools.generate_testcase(model, [v, w])

    testtools.generate_testcase(model, [v, w], backprop=True)


if __name__ == '__main__':
    main()
