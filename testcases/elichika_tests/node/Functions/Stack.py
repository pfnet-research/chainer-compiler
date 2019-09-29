# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools
import numpy as np

class Stack(chainer.Chain):
    def forward(self, x, y):
        y1 = F.stack((x, y))
        y2 = np.stack([x, y])
        return y1, y2


class StackAxis0(chainer.Chain):
    def forward(self, x, y):
        y1 = F.stack((x, y), axis=0)
        y2 = np.stack([x, y], axis=0)
        return y1, y2


class StackAxis1(chainer.Chain):
    def forward(self, x, y):
        y1 = F.stack((x, y), axis=1)
        y2 = np.stack([x, y], axis=1)
        return y1, y2


class StackAxis2(chainer.Chain):
    def forward(self, x, y):
        y1 = F.stack((x, y), axis=2)
        y2 = np.stack([x, y], axis=2)
        return y1, y2


# ======================================

import numpy as np

def main():
    v = np.random.rand(5, 4, 2).astype(np.float32)
    w = np.random.rand(5, 4, 2).astype(np.float32)

    testtools.generate_testcase(Stack, [v, w])
    testtools.generate_testcase(StackAxis0, [v, w], subname='axis0')
    testtools.generate_testcase(StackAxis1, [v, w], subname='axis1')
    testtools.generate_testcase(StackAxis2, [v, w], subname='axis2')

if __name__ == '__main__':
    main()
