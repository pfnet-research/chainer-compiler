# coding: utf-8

import chainer


class A(chainer.Chain):
    def forward(self, xs, xss, ps):
        y1 = xs[2]
        y2 = xs[3:5]
        y3 = xs[ps[0]]
        y4 = xs[ps[0]:ps[0]+2]
        y5 = xss[ps[0]:10, ps[1]:ps[1]+4]
        y6 = xss[ps[0], ps[1]:ps[1]+4]
        y7 = xss[3, ps[0]]
        y8 = xs[-1]
        y9 = xs[-2]
        # TODO(satos) listによるインデクシングもできるようにする
        # y10 = xs[[1,3,5]]
        return y1, y2, y3, y4, y5, y6, y7, y8, y9


class ListSlice(chainer.Chain):
    def forward(self, x):
        # Use `shape` to make a sequence.
        xs = x.shape
        y1 = np.array(xs[2])
        y2 = np.array(xs[-2])
        y3 = np.array(xs[:2])
        y4 = np.array(xs[1:3])
        y5 = np.array(xs[1::2])
        return y1, y2, y3, y4, y5


class SliceStep(chainer.Chain):
    def forward(self, xs):
        return xs[1:6:2]


class SliceStepSecond(chainer.Chain):
    def forward(self, xs):
        return xs[:, 1:-2:2]


class SliceAll(chainer.Chain):
    def forward(self, xs):
        return xs[:]


class SliceAllSecond(chainer.Chain):
    def forward(self, xs):
        return xs[:, :]


# ======================================


import testtools
import numpy as np


def main():
    n_maxlen = 10

    model = A()

    u = np.random.rand(n_maxlen+6).astype(np.float32)
    v = np.random.rand(n_maxlen+6, n_maxlen+6).astype(np.float32)
    w = np.random.randint(0, n_maxlen, size=2)

    testtools.generate_testcase(model, [u, v, w])

    x = np.random.rand(7, 5, 3, 4)
    testtools.generate_testcase(ListSlice(), [x], subname='list')

    testtools.generate_testcase(SliceStep(), [x], subname='step')
    testtools.generate_testcase(SliceStepSecond(), [x], subname='step_second')

    testtools.generate_testcase(SliceAll(), [x], subname='all')
    testtools.generate_testcase(SliceAllSecond(), [x], subname='all_second')

if __name__ == '__main__':
    main()
