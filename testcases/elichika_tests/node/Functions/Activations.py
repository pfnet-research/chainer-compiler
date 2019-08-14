# coding: utf-8

import numpy as np
from chainer_compiler.elichika import testtools
import chainer
import chainer.functions as F


class Simple(chainer.Chain):

    def __init__(self, func):
        super(Simple, self).__init__()
        self.func = func

    def forward(self, x):
        y = self.func(x)
        return y


class Param1(chainer.Chain):
    def __init__(self, func, v1):
        super(Param1, self).__init__()
        self.func = func
        self.v1 = v1

    def forward(self, x):
        y = self.func(x, self.v1)
        return y


class Param2(chainer.Chain):
    def __init__(self, func, v1, v2):
        super(Param2, self).__init__()
        self.func = func
        self.v1 = v1
        self.v2 = v2

    def forward(self, x):
        y = self.func(x, self.v1,  self.v2)
        return y

# ======================================


def main():
    x = np.random.rand(6, 4, 3).astype(np.float32) - 0.5

    simple_targets = [
        'clipped_relu',
        'leaky_relu',
        'log_softmax',
        'elu',
        'relu',
        'selu',
        'sigmoid',
        'softmax',
    ]

    for target in simple_targets:
        func = getattr(F, target)
        testtools.generate_testcase(
            Simple(func), [x], subname=target+'_simple')

    param1_targets = [
        ('clipped_relu', 10.0),
        ('leaky_relu', 0.1),
        ('log_softmax', 2),
        ('elu', 0.9),
        ('selu', 2.0),
        ('softmax', 2),
    ]

    for target in param1_targets:
        func = getattr(F, target[0])
        testtools.generate_testcase(
            Param1(func, target[1]), [x], subname=target[0]+'_param1')

    param2_targets = [
        ('selu', 2.0, 2.0),
    ]

    for target in param2_targets:
        func = getattr(F, target[0])
        testtools.generate_testcase(
            Param2(func, target[1], target[2]), [x], subname=target[0]+'_param2')


if __name__ == '__main__':
    main()
