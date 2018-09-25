# coding: utf-8

import chainer


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs, xss, ps):
        y1 = xs[2]
        y2 = xs[3:5]
        y3 = xs[ps[0]]
        y4 = xs[ps[0]:ps[0]+2]
        y5 = xss[ps[0]:10, ps[1]:ps[1]+4]
        y6 = xss[ps[0], ps[1]:ps[1]+4]
        y7 = xss[3, ps[0]]
        # TODO(satos) listによるインデクシングもできるようにする
        # y8 = xs[[1,3,5]]
        return y1, y2, y3, y4, y5, y6, y7

# ======================================


import ch2o
import numpy as np


if __name__ == '__main__':
    n_maxlen = 10

    model = A()

    u = np.random.rand(n_maxlen+6).astype(np.float32)
    v = np.random.rand(n_maxlen+6, n_maxlen+6).astype(np.float32)
    w = np.random.randint(0, n_maxlen, size=2)

    ch2o.generate_testcase(model, [u, v, w])
