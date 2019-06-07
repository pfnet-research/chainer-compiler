# coding: utf-8

import chainer
import chainer.functions as F
import numpy as np


class Separate(chainer.Chain):
    def forward(self, x):
        return list(F.separate(x))


class SeparateAxis0(chainer.Chain):
    def forward(self, x):
        return list(F.separate(x, axis=0))


class SeparateAxis1(chainer.Chain):
    def forward(self, x):
        return list(F.separate(x, axis=1))


class SeparateAxis2(chainer.Chain):
    def forward(self, x):
        return list(F.separate(x, axis=2))


# ======================================


import chainer_compiler.ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(12)

    x = np.random.rand(2, 3, 4).astype(np.float32)

    ch2o.generate_testcase(Separate, [x])
    ch2o.generate_testcase(SeparateAxis0, [x], subname='axis_0')
    ch2o.generate_testcase(SeparateAxis1, [x], subname='axis_1')
    ch2o.generate_testcase(SeparateAxis2, [x], subname='axis_2')
