# coding: utf-8

from chainer_compiler.elichika import testtools
import chainer
import chainer.functions as F
import numpy as np


class SplitAxis(chainer.Chain):
    def forward(self, xs, ilens):
        y1 = F.split_axis(xs, ilens, axis=0)
        # convert to tuple into list
        y1 = [x for x in y1]
        return y1


class SplitAxis1(chainer.Chain):
    def forward(self, xs, ilens):
        y1 = F.split_axis(xs, ilens, axis=1)
        # convert to tuple into list
        y1 = [x for x in y1]
        return y1


class SplitAxisSections(chainer.Chain):
    def forward(self, xs):
        y1 = F.split_axis(xs, 2, 0)
        # convert to tuple into list
        y1 = [x for x in y1]
        return y1


class SplitAxisSections1(chainer.Chain):
    def forward(self, xs):
        y1 = F.split_axis(xs, 2, axis=1)
        # convert to tuple into list
        y1 = [x for x in y1]
        return y1

# ======================================


def main():
    import numpy as np
    np.random.seed(12)

    xs = np.random.rand(20, 20).astype(np.float32)
    ilens = [1, 3, 5, 8, 14]

    testtools.generate_testcase(SplitAxis, [xs, ilens])
    testtools.generate_testcase(SplitAxis1, [xs, ilens], subname='axis1')
    testtools.generate_testcase(SplitAxisSections, [xs], subname='sections')
    testtools.generate_testcase(SplitAxisSections1, [xs], subname='sections_axis1')


if __name__ == '__main__':
    main()
