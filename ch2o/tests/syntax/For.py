# coding: utf-8

import chainer
import chainer.functions as F


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


# ======================================


import ch2o
import numpy as np


if __name__ == '__main__':
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
