# coding: utf-8

import chainer
import chainer.functions as F


class Size(chainer.Chain):
    def forward(self, x):
        y1 = x.size
        return y1


# ======================================

import testtools
import numpy as np

def main():
    model = Size()

    x = np.random.rand(12, 6, 4).astype(np.float32)
    testtools.generate_testcase(model, [x])


if __name__ == '__main__':
    main()
