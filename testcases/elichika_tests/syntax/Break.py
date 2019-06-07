# coding: utf-8

import chainer
import chainer.functions as F


class Break(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            if i == y:
                continue
            ret = ret + i
        return ret


class BreakNested(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            for j in range(x):
                if i == y:
                    if j == y:
                        continue
                ret = ret + j
        return ret


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np


def main():
    x, y = 10, 5
    testtools.generate_testcase(Break, [x, y], subname='break')
    testtools.generate_testcase(BreakNested, [x, y], subname='break_nested')


if __name__ == '__main__':
    main()
