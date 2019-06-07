# coding: utf-8

import chainer
import chainer.functions as F
import numpy as np


class SplitAxis(chainer.Chain):
    def forward(self, xs, ilens):
        y1 = F.split_axis(xs, ilens, axis=0)
        # この時点でTuple!! なのでrange based for でlistにする
        y1 = [x for x in y1]
        return y1


class SplitAxis1(chainer.Chain):
    def forward(self, xs, ilens):
        y1 = F.split_axis(xs, ilens, axis=1)
        # この時点でTuple!! なのでrange based for でlistにする
        y1 = [x for x in y1]
        return y1


class SplitAxisSections(chainer.Chain):
    def forward(self, xs):
        y1 = F.split_axis(xs, 2, 0)
        # この時点でTuple!! なのでrange based for でlistにする
        y1 = [x for x in y1]
        return y1


class SplitAxisSections1(chainer.Chain):
    def forward(self, xs):
        y1 = F.split_axis(xs, 2, axis=1)
        # この時点でTuple!! なのでrange based for でlistにする
        y1 = [x for x in y1]
        return y1

# ======================================


from chainer_compiler import ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)

    xs = np.random.rand(20, 20).astype(np.float32)
    ilens = [1, 3, 5, 8, 14]

    ch2o.generate_testcase(SplitAxis, [xs, ilens])
    ch2o.generate_testcase(SplitAxis1, [xs, ilens], subname='axis1')
    ch2o.generate_testcase(SplitAxisSections, [xs], subname='sections')
    ch2o.generate_testcase(SplitAxisSections1, [xs], subname='sections_axis1')
