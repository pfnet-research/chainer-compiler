# coding: utf-8

import chainer
import chainer.functions as F

# Network definition


class Sigmoid(chainer.Chain):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        y1 = F.sigmoid(x)
        return y1


# ======================================
import testtools
import numpy as np

def main():
    model = Sigmoid()

    np.random.seed(314)

    x = np.random.rand(6, 4).astype(np.float32)
    testtools.generate_testcase(model, [x])


if __name__ == '__main__':
    main()
