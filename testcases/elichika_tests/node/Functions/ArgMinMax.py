# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer_compiler.elichika import testtools

class ArgMin(chainer.Chain):
    def forward(self, v1):
        return F.argmin(v1)

class ArgMinNumpy(chainer.Chain):
    def forward(self, v1):
        return np.argmin(v1)

class ArgMinAxis(chainer.Chain):
    def forward(self, v1):
        return F.argmin(v1,axis=1)

class ArgMinAxisNumpy(chainer.Chain):
    def forward(self, v1):
        return np.argmin(v1,axis=1)

class ArgMax(chainer.Chain):
    def forward(self, v1):
        return F.argmax(v1)

class ArgMaxNumpy(chainer.Chain):
    def forward(self, v1):
        return np.argmax(v1)

class ArgMaxAxis(chainer.Chain):
    def forward(self, v1):
        return F.argmax(v1,axis=1)

class ArgMaxAxisNumpy(chainer.Chain):
    def forward(self, v1):
        return np.argmax(v1,axis=1)

# ======================================

def main():
    np.random.seed(314)
    a1 = np.random.rand(6, 2, 3).astype(np.float32)

    testtools.generate_testcase(ArgMin(), [a1], subname='argmin')
    testtools.generate_testcase(ArgMinNumpy(), [a1], subname='argmin_np')
    testtools.generate_testcase(ArgMinAxis(), [a1], subname='argmin_axis')
    testtools.generate_testcase(ArgMinAxisNumpy(), [a1], subname='argmin_axis_np')

    testtools.generate_testcase(ArgMax(), [a1], subname='argmax')
    testtools.generate_testcase(ArgMaxNumpy(), [a1], subname='argmax_np')
    testtools.generate_testcase(ArgMaxAxis(), [a1], subname='argmax_axis')
    testtools.generate_testcase(ArgMaxAxisNumpy(), [a1], subname='argmax_axis_np')

if __name__ == '__main__':
    main()
    