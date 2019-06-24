# coding: utf-8

import chainer
import chainer.functions as F


class Return(chainer.Chain):
    def forward(self, x, y):
        for i in range(x):
            if i == y:
                return i
        return 0


class ReturnNested(chainer.Chain):
    def forward(self, x, y):
        def func(a, b):
            for i in range(a):
                if i == b:
                    return i + 1
            return 0
        z = func(x, y)
        for i in range(z):
            if i == y:
                return i
        return 0

class ReturnContinueMixed(chainer.Chain):
    def forward(self, x, y, z):
        ret = 0
        for i in range(x):
            if i == y:
                continue
            for j in range(x):
                if j == z:
                    continue
                if j == y:
                    return ret
                ret += i * j
        return ret

class ReturnContinueBreakMixed(chainer.Chain):
    def forward(self, x, y, z):
        ret = 0
        for i in range(x):
            if i == y:
                continue
            for j in range(x):
                if j == z:
                    break
                if j == y:
                    return ret
                ret += i * j
        return ret


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np


def main():
    x, y, z = 10, 5, 4
    testtools.generate_testcase(Return, [x, y], subname='return')
    # testtools.generate_testcase(ReturnNested, [x, y], subname='return_nested')  #TODO: Nested function declaration is not supported.
    testtools.generate_testcase(ReturnContinueMixed, [x, y, z], subname='return_continue_mixed')
    testtools.generate_testcase(ReturnContinueBreakMixed, [x, y, z], subname='return_continue_break_mixed')


if __name__ == '__main__':
    main()
