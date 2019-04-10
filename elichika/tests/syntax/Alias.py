# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class Add(chainer.Chain):
    def forward(self, x):
        y = x
        y += 3
        return x


class If(chainer.Chain):
    def forward(self, x, cond):
        y = x
        if cond:
            y += 3
        else:
            y += 10
        return x


# ======================================

import testtools
import numpy as np


def main():
    testtools.generate_testcase(Add(), [42], subname='add_int')

    testtools.generate_testcase(Add(), [np.array(42)], subname='add_np')

    testtools.generate_testcase(If(), [42, False],
                                subname='if_false_int')
    testtools.generate_testcase(If(), [42, True],
                                subname='if_true_int')
    # TODO(hamaji): Enable this tests once they are fixed.
    # testtools.generate_testcase(If(), [np.array(42), False],
    #                             subname='if_false_np')
    # testtools.generate_testcase(If(), [np.array(42), True],
    #                             subname='if_true_np')


if __name__ == '__main__':
    main()
