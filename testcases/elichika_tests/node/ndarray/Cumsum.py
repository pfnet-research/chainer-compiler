# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer_compiler.elichika import testtools

class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, ps):
        y1 = np.cumsum(ps)
        return y1

# ======================================

def main():
    import numpy as np
    np.random.seed(12)

    model = A()

    ps = [3, 1, 4, 1, 5, 9, 2]

    testtools.generate_testcase(model, [ps])

if __name__ == '__main__':
    main()