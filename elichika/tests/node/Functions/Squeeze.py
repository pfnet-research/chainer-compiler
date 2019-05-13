# coding: utf-8

import chainer
import chainer.functions as F
import testtools

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

import numpy as np

def main():
    x = np.random.rand(3, 1, 4, 1, 5, 1).astype(np.float32)

    testtools.generate_testcase(Squeeze(), [x])
    testtools.generate_testcase(SqueezeAxes(), [x], subname='axes')
    testtools.generate_testcase(SqueezeNoAxes(), [x], subname='noaxes')

if __name__ == '__main__':
    main()
