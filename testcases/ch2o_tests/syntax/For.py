# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class A(chainer.Chain):
    def forward(self, xs, p):
        v = []
        for i in range(p):
            v.append(xs[:i])
        return v


class B(chainer.Chain):
    def forward(self, xs, l):
        inputs = F.pad_sequence(xs)
        h = inputs[:, 0]
        for time in range(l):
            h = inputs[:, time]
        return h


class C(chainer.Chain):
    def forward(self):
        bs = 0
        cs = 0
        for i in range(4):
            cs = i
            bs = cs
        return bs, cs


class D(chainer.Chain):
    def forward(self):
        for i in range(4):
            o = i
        return o


class UpdateSelf(chainer.Chain):
    def forward(self, x):
        self.x = x
        for i in range(5):
            self.x += i
        return self.x


class UpdateSelfLiteral(chainer.Chain):
    def forward(self):
        self.x = 42
        for i in range(5):
            self.x += i
        return self.x


class UpdateSelfLiteralInInit(chainer.Chain):
    def __init__(self):
        super(UpdateSelfLiteralInInit, self).__init__()
        self.x = 42

    def forward(self):
        for i in range(5):
            self.x += i
        return self.x


class ForBackprop(chainer.Chain):
    def __init__(self):
        super(ForBackprop, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, 3)

    def forward(self, x, n):
        for i in range(n):
            x = self.l(x)
        return x


class DoubleForBackprop(chainer.Chain):
    def __init__(self):
        super(DoubleForBackprop, self).__init__()
        with self.init_scope():
            self.l = L.Linear(None, 3)

    def forward(self, x, n, m):
        for i in range(n):
            for j in range(m):
                x = self.l(x)
        return x


# ======================================


from chainer_compiler import ch2o
import numpy as np


if __name__ == '__main__':
    np.random.seed(42)

    model = A()

    v = np.random.rand(10).astype(np.float32)
    p = np.int64(5)
    ch2o.generate_testcase(model, [v, p])

    model = B()
    length = 4
    xs = []
    for i in range(length):
        xs.append(np.random.rand(length, 5).astype(dtype=np.float32))
    args = [xs, length]
    ch2o.generate_testcase(model, args, subname='closure_bug')

    ch2o.generate_testcase(C(), [], subname='multi_ref')

    ch2o.generate_testcase(D(), [], subname='leak')

    ch2o.generate_testcase(UpdateSelf(), [42], subname='update_self')

    ch2o.generate_testcase(UpdateSelfLiteral(), [],
                           subname='update_self_literal')

    ch2o.generate_testcase(UpdateSelfLiteralInInit, [],
                           subname='update_self_literal_in_init')

    ch2o.generate_testcase(ForBackprop,
                           [np.random.rand(4, 3).astype(np.float32), 2],
                           subname='for', backprop=True)

    ch2o.generate_testcase(DoubleForBackprop,
                           [np.random.rand(4, 3).astype(np.float32), 2, 5],
                           subname='double_for', backprop=True)
