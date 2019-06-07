# coding: utf-8

import chainer


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs, ps, p):
        t = [xs[:i+1] for i in range(p)]
        y1 = [v[-2:-1] for v in t]
        v = []
        v.append(xs[2])
        v.append(xs[0])
        return y1, v

# ======================================


import chainer_compiler.ch2o
import numpy as np

if __name__ == '__main__':
    model = A()

    wn = 1
    v = np.random.rand(10).astype(np.float32)
    w = np.random.randint(0, 5, size=wn)
    p = np.int64(wn)
    ch2o.generate_testcase(model, [v, w, p])
