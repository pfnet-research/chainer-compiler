# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import testtools

class Mean(chainer.Chain):
    def forward(self, x):
        return F.mean(x, axis=1)


class MeanKeepdims(chainer.Chain):
    def forward(self, x):
        return F.mean(x, axis=0, keepdims=True)


class MeanTupleAxis(chainer.Chain):
    def forward(self, x):
        return F.mean(x, axis=(1, 2))


class MeanAllAxis(chainer.Chain):
    def forward(self, x):
        return F.mean(x)


# ======================================

def main():
    np.random.seed(314)
    a = np.random.rand(3, 5, 4).astype(np.float32)

    testtools.generate_testcase(Mean(), [a])

    testtools.generate_testcase(MeanKeepdims(), [a], subname='keepdims')

    testtools.generate_testcase(MeanTupleAxis(), [a], subname='tuple_axis')

    testtools.generate_testcase(MeanAllAxis(), [a], subname='all_axis')

if __name__ == '__main__':
    main()
