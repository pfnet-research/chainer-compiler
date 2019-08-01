# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer_compiler.elichika import testtools

class Sum(chainer.Chain):
    def forward(self, x):
        return F.sum(x, axis=1)


class SumSelf(chainer.Chain):
    def forward(self, x):
        return x.sum(axis=1)

class SumKeepdims(chainer.Chain):
    def forward(self, x):
        return F.sum(x, axis=0, keepdims=True)


class SumTupleAxis(chainer.Chain):
    def forward(self, x):
        return F.sum(x, axis=(1, 2))


class SumAllAxis(chainer.Chain):
    def forward(self, x):
        return F.sum(x)


class SumBackprop(chainer.Chain):
    def __init__(self, axis, keepdims):
        super(SumBackprop, self).__init__()
        self.axis = axis
        self.keepdims = keepdims
        with self.init_scope():
            self.l = L.Linear(None, 5)

    def forward(self, x):
        x = self.l(x)
        x = F.reshape(x, (3, 2, 5))
        x = F.sum(x, axis=self.axis, keepdims=self.keepdims)
        return x


# ======================================

def main():
    np.random.seed(314)
    a = np.random.rand(6, 2, 3).astype(np.float32)

    testtools.generate_testcase(Sum(), [a])

    testtools.generate_testcase(SumSelf(), [a], subname='self')

    testtools.generate_testcase(SumKeepdims(), [a], subname='keepdims')

    testtools.generate_testcase(SumTupleAxis(), [a], subname='tuple_axis')

    testtools.generate_testcase(SumAllAxis(), [a], subname='all_axis')

    for axis in [0, 1, 2, (0, 2), (1, 2), None]:
        for keepdims in [False, True]:
            name = 'axis%s_kd%s' % (axis, keepdims)
            testtools.generate_testcase(lambda: SumBackprop(axis, keepdims), [a],
                                   subname=name, backprop=True)


if __name__ == '__main__':
    main()
