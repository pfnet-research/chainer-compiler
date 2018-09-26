# coding: utf-8

import chainer


class A(chainer.Chain):
    def forward(self, xs, ps, p):
        y1 = [xs[x, x+2] for x in range(p)]
        y2 = [xs[ps[x], ps[x]+3] for x in range(p)]
        return y1, y2


class B(chainer.Chain):
    def forward(self, x):
        return range(x)


# ======================================


import ch2o
import numpy as np

if __name__ == '__main__':
    model = A()

    wn = 5
    v = np.random.rand(10, 20).astype(np.float32)
    w = np.random.randint(0, 5, size=wn)
    p = np.int64(wn)
    ch2o.generate_testcase(model, [v, w, p])

    ch2o.generate_testcase(B(), [5], subname='stop')
