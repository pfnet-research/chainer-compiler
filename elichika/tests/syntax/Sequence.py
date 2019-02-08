# coding: utf-8

import chainer

class Basic(chainer.Chain):
    def __init__(self):
        super(Basic, self).__init__()

    def forward(self, v):
        x = [v]
        return x


class Index(chainer.Chain):
    def __init__(self):
        super(Index, self).__init__()

    def forward(self, v):
        x = [v,v+1,v+2,v+3]
        return x[1]

class Slice(chainer.Chain):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, v):
        x = [v,v+1,v+2,v+3]
        return x[1:2]

class Append(chainer.Chain):
    def __init__(self):
        super(Append, self).__init__()

    def forward(self, v):
        x = []
        x.append(v)
        x.append(v)
        return x

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


import testtools
import numpy as np


def main():
    testtools.generate_testcase(Basic(), [10], subname='Basic')

    testtools.generate_testcase(Index(), [10], subname='Index')

    testtools.generate_testcase(Slice(), [10], subname='Slice')

    testtools.generate_testcase(Append(), [10], subname='Append')

    model = A()

    wn = 1
    v = np.random.rand(10).astype(np.float32)
    w = np.random.randint(0, 5, size=wn)
    p = np.int64(wn)
    testtools.generate_testcase(model, [v, w, p], subname='A')


if __name__ == '__main__':
    main()
