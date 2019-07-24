# coding: utf-8

import chainer
import chainer.links as L
from chainer import serializers

class ConvND(chainer.Chain):

    def __init__(self, ndim, nobias):
        super(ConvND, self).__init__()
        with self.init_scope():
            self.l1 = L.ConvolutionND(ndim, 7, 10, 3,
                                      stride=1, pad=1, nobias=nobias)

    def forward(self, x):
        y1 = self.l1(x)
        return y1

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np


def main():
    np.random.seed(123)
    x = np.random.rand(2, 7, 15, 17).astype(np.float32)

    for ndim in [1, 2, 3]:
        shape = (2, 7) + tuple(15 + i * 2 for i in range(ndim))
        x = np.random.rand(*shape).astype(np.float32)
        for nobias in [False, True]:
            subname = '%dd' % ndim
            if nobias:
                subname += '_nobias'

            model = ConvND(ndim, nobias)
            testtools.generate_testcase(model, [x], subname=subname)


if __name__ == '__main__':
    main()
