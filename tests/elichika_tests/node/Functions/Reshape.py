# coding: utf-8

import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools
import numpy as np

# Network definition


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x, p):
        y1 = F.reshape(x, (24, 12))
        y2 = F.reshape(x, (6, -1, 12))
        y3 = F.reshape(x, (-1, p))
        return (y1, y2, y3)


# ======================================

def main():
    import numpy as np
    np.random.seed(314)

    model = A()

    x = np.random.rand(12, 6, 4).astype(np.float32)
    p = np.int64(3)
    testtools.generate_testcase(model, [x, p])

if __name__ == '__main__':
    main()
    