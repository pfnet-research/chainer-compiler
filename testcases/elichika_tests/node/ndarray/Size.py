# coding: utf-8

import chainer
import chainer.functions as F

from chainer_compiler.elichika import testtools
import numpy as np


class Size(chainer.Chain):
    def forward(self, x):
        y1 = x.size
        return y1


# ======================================


def main():
    import numpy as np
    np.random.seed(314)

    x = np.random.rand(12, 6, 4).astype(np.float32)

    testtools.generate_testcase(Size(), [x])


if __name__ == '__main__':
    main()
