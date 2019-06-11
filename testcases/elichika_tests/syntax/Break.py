# coding: utf-8

import chainer
import chainer.functions as F


class Break(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            if i == y:
                break
            ret = ret + i
        return ret


class BreakNested(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            if i == y - 1:
                break
            for j in range(x):
                if i == y:
                    if j == y:
                        break
                ret = ret + j
        return ret

class BreakContinueMixed(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            if i == y:
                break
            if i == y - 1:
                continue
            ret = ret + i
        return ret


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np


def main():
    x, y = 10, 5
    testtools.generate_testcase(Break, [x, y], subname='break')
    testtools.generate_testcase(BreakNested, [x, y], subname='break_nested')
    testtools.generate_testcase(BreakContinueMixed, [x, y], subname='break_continue_mixed')


if __name__ == '__main__':
    main()
