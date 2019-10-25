# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools
import numpy as np


class A(chainer.Chain):

    def forward(self, x):
        y1 = F.transpose(x, (0, 2, 1))
        y2 = F.transpose(x, (1, 0, 2))
        y3 = F.transpose(x, (2, 1, 0))
        y4 = x.transpose((0, 2, 1))
        y5 = np.transpose(x, (1, 0, 2))
        y6 = x.transpose((2, 1, 0))
        return (y1, y2, y3, y4, y5, y6)


# ======================================

def main():
    import numpy as np
    np.random.seed(314)

    model = A()

    x = np.random.rand(12, 6, 4).astype(np.float32)
    testtools.generate_testcase(model, [x])

if __name__ == '__main__':
    main()
    