# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class BroadcastTo(chainer.Chain):
    def forward(self, x):
        y1 = F.broadcast_to(x, (2, 6, 4, 3))
        return y1


class BroadcastToBackprop(chainer.Chain):
    def __init__(self):
        super(BroadcastToBackprop, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 5)

    def forward(self, x):
        x = self.l1(x)
        x = F.reshape(x, (6, 5, 1))
        y = F.broadcast_to(x, (2, 6, 5, 3))
        return y


# ======================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    np.random.seed(314)

    x = np.random.rand(6, 4, 1).astype(np.float32) - 0.5
    testtools.generate_testcase(BroadcastTo(), [x])

    x = np.random.rand(6, 3).astype(np.float32) - 0.5
    testtools.generate_testcase(BroadcastToBackprop(), [x], backprop=True)


if __name__ == '__main__':
    main()
