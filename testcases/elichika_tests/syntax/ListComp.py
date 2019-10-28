# coding: utf-8

import chainer

class Basic1(chainer.Chain):
    def forward(self):
        r = range(2)
        x = [y for y in r]
        return x

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

class B(chainer.Chain):

    def __init__(self):
        super(B, self).__init__()

    def forward(self, xs):
        y = [x.shape for x in xs]
        return y[0]

# ======================================


from chainer_compiler.elichika import testtools
import numpy as np


def main():
    np.random.seed(314)

    testtools.generate_testcase(Basic1(), [], subname='Basic1')

    model = A()

    v = np.random.rand(10, 20).astype(np.float32)
    ps = np.array([3, 4])
    qs = np.array([1, 2, 3, 4, 5])
    p = np.int64(5)
    testtools.generate_testcase(model, [v, ps, p, qs], subname='A')
    # testtools.generate_testcase(B(), [v], subname='listcomp_bug')

if __name__ == '__main__':
    main()
