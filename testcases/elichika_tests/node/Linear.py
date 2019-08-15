# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers

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

    def __init__(self, num_hidden):
        super(B, self).__init__()
        with self.init_scope():
            self.l = L.Linear(num_hidden * 2, num_hidden * 4)

    def forward(self, xs, h):
        inputs = F.pad_sequence(xs)
        gate = self.l(F.concat((inputs[:, 0], h), axis=1))
        return gate

class Axes(chainer.Chain):

    def __init__(self):
        super(Axes, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(3, 5)

    def forward(self, x):
        y1 = self.l1(x, n_batch_axes=2)
        return y1

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    np.random.seed(314)

    model = A(3)
    x = np.random.rand(5, 7).astype(np.float32)
    testtools.generate_testcase(model, [x])

    y = np.random.rand(2, 4, 3).astype(np.float32)
    testtools.generate_testcase(Axes(), [y], subname='axes')

    testtools.generate_testcase(Axes(), [y], subname='axes', backprop=True)

    # Value mismatch bug.
    num_hidden = 5
    model = lambda: B(num_hidden)
    xs = []
    for l in [4, 3, 2]:
        xs.append(np.random.rand(l, num_hidden).astype(dtype=np.float32))
    h = np.zeros((3, num_hidden), dtype=np.float32)

    testtools.generate_testcase(model, [xs, h], subname='none_param')


if __name__ == '__main__':
    main()
