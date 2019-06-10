# coding: utf-8

import chainer
import chainer.functions as F


class Squeeze(chainer.Chain):
    def forward(self, x):
        return F.squeeze(x, 1)


class SqueezeAxes(chainer.Chain):
    def forward(self, x):
        return F.squeeze(x, axis=(1, 3))


class SqueezeNoAxes(chainer.Chain):
    def forward(self, x):
        return F.squeeze(x)


# ======================================

from chainer_compiler import ch2o
import numpy as np

if __name__ == '__main__':
    x = np.random.rand(3, 1, 4, 1, 5, 1).astype(np.float32)

    ch2o.generate_testcase(Squeeze(), [x])
    ch2o.generate_testcase(SqueezeAxes(), [x], subname='axes')
    ch2o.generate_testcase(SqueezeNoAxes(), [x], subname='noaxes')
