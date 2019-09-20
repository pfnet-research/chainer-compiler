# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer_compiler.elichika import testtools

class Simple(chainer.Chain):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, v):
        return self.func(v)

class Axis(chainer.Chain):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, v):
        return self.func(v, axis=1)

class KeepDims(chainer.Chain):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, v):
        return self.func(v, keepdims=True)

class AxisKeepDims(chainer.Chain):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, v):
        return self.func(v, axis=1, keepdims=True)

# ======================================

def main():
    np.random.seed(314)
    a1 = np.random.rand(6, 2, 3).astype(np.float32)
 
    def test(func, name):
        testtools.generate_testcase(Simple(func), [a1], subname= name + '_simple')
        testtools.generate_testcase(Axis(func), [a1], subname= name + '_axis')
        testtools.generate_testcase(KeepDims(func), [a1], subname= name + '_keepdims')
        testtools.generate_testcase(AxisKeepDims(func), [a1], subname= name + '_axiskeepdims')

    test(F.min, 'min')
    test(F.max, 'max')

if __name__ == '__main__':
    main()