# coding: utf-8

import chainer
import chainer.functions as F


class ExpandDims(chainer.Chain):

    def __init__(self):
        super(ExpandDims, self).__init__()

    def forward(self, x):
        y = F.expand_dims(x, axis=1)
        y2 = F.expand_dims(x, 1)
        return y, y2


# ======================================

from chainer_compiler.elichika import testtools

def main():
    import numpy as np
    np.random.seed(314)
    model = ExpandDims()

    x = np.random.rand(6, 4).astype(np.float32) - 0.5
    testtools.generate_testcase(model, [x])


if __name__ == '__main__':
    main()
