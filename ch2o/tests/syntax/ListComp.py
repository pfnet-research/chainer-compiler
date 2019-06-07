# coding: utf-8

import chainer


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, xs, ps, p, qs):
        # y5 = [xs[i][:p] for p in ps for i in range(p)]
        # y5 = [p for j in ps for i in ps]
        """
        y5 = []
        for j in ps:
            for i in qs:
                y5.append(p)
        return y5
        """
        y1 = [x[3:5] for x in xs]
        y2 = [x[ps[0]:ps[0]+3] for x in xs]
        y3 = [x[p:p+4] for x in xs]
        y4 = [xs[i][:i] for i in range(p)]
        y5 = [3 for x in xs]
        return y1, y2, y3, y4, y5

# ======================================


import chainer_compiler.ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    model = A()

    v = np.random.rand(10, 20).astype(np.float32)
    ps = np.array([3, 4])
    qs = np.array([1, 2, 3, 4, 5])
    p = np.int64(5)
    ch2o.generate_testcase(model, [v, ps, p, qs])
