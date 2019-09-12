# coding: utf-8

from chainer_compiler.elichika import testtools
import numpy as np
import chainer
import chainer.functions as F


class Array(chainer.Chain):
    def forward(self):
        y1 = np.array([4.0, 2.0, 3.0], dtype=np.float32)
        return y1


class ArrayCast(chainer.Chain):
    def forward(self):
        y1 = np.array([4.0, 2.0, 3.0], dtype=np.int32)
        return y1


class Array2D(chainer.Chain):
    def forward(self):
        y1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        return y1


# ======================================


def main():
    import numpy as np
    np.random.seed(314)

    testtools.generate_testcase(Array, [], subname='default')
    testtools.generate_testcase(ArrayCast, [], subname='cast')
    # TODO(hamaji): Fix sequence of sequences.
    # testtools.generate_testcase(Array2D, [], subname='2d')


if __name__ == '__main__':
    main()
