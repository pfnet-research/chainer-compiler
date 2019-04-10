# coding: utf-8

import numpy as np
import testtools
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


class SeqAdd(chainer.Chain):
    def forward(self, x, a):
        y = x
        y += a
        return x


# ======================================


def main():
    testtools.generate_testcase(Add(), [42], subname='add_int')

    testtools.generate_testcase(Add(), [np.array(42)], subname='add_np')

    testtools.generate_testcase(If(), [42, False],
                                subname='if_false_int')
    testtools.generate_testcase(If(), [42, True],
                                subname='if_true_int')

    testtools.generate_testcase(If(), [np.array(42), False],
                                subname='if_false_np')
    testtools.generate_testcase(If(), [np.array(42), True],
                                subname='if_true_np')

    # TODO(hamaji): Enable this tests once they are fixed.
    # testtools.generate_testcase(SeqAdd(), [[42], [3]], subname='add_list')

    # testtools.generate_testcase(SeqAdd(), [(42,), (3,)], subname='add_tuple')


if __name__ == '__main__':
    main()
