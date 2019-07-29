# coding: utf-8

import chainer


class SumMinMax(chainer.Chain):

    def __init__(self):
        super(SumMinMax, self).__init__()

    def forward(self, x, y, z, xs):
        ret = 1
        ret *= sum(xs)
        ret += max(x, y, z)
        ret -= min(x, y, z)
        ret += max(xs)
        ret -= min(xs)
        return ret

# ======================================

from chainer_compiler.elichika import testtools
import numpy as np

def main():
    np.random.seed(314)

    # TODO(rchouras): After the `Sum`, `Max` and `Min` ONNX nodes are fixed, uncomment the tests. 
    # testtools.generate_testcase(SumMinMax(), [2, 1, 3, [3, 5, 4]], subname='basic') 
    # testtools.generate_testcase(SumMinMax(), [np.array(2), 1, 3, np.array([3, 5, 4])], subname='with_tensor')


if __name__ == '__main__':
    main()
