# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer_compiler.elichika import testtools

class Minimum(chainer.Chain):
    def forward(self, v1,v2):
        return F.minimum(v1, v2)


class MinimumNumpy(chainer.Chain):
    def forward(self, v1,v2):
        return np.minimum(v1, v2)

# ======================================

def main():
    np.random.seed(314)
    a1 = np.random.rand(6, 2, 3).astype(np.float32)
    a2 = np.random.rand(6, 2, 3).astype(np.float32)

    testtools.generate_testcase(Minimum(), [a1, a2])
    testtools.generate_testcase(MinimumNumpy(), [a1, a2], subname='np')

if __name__ == '__main__':
    main()