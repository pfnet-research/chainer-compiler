# coding: utf-8

import chainer


class Range(chainer.Chain):
    def forward(self, x):
        return range(x)


class RangeStop(chainer.Chain):
    def forward(self, x, y):
        return range(x, y)


class RangeStep(chainer.Chain):
    def forward(self, x, y, z):
        return range(x, y, z)


class RangeListComp(chainer.Chain):
    def forward(self, xs, ps, p):
        y1 = [xs[x, x+2] for x in range(p)]
        y2 = [xs[ps[x], ps[x]+3] for x in range(p)]
        return y1, y2


# ======================================


import ch2o
import numpy as np

if __name__ == '__main__':
    ch2o.generate_testcase(Range, [5])

    ch2o.generate_testcase(RangeStop(), [5, 8], subname='stop')

    ch2o.generate_testcase(RangeStep(), [5, 19, 2], subname='step')

    wn = 5
    v = np.random.rand(10, 20).astype(np.float32)
    w = np.random.randint(0, 5, size=wn)
    p = np.int64(wn)
    ch2o.generate_testcase(RangeListComp, [v, w, p], subname='list_comp')
