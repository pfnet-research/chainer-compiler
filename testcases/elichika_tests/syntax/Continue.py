# coding: utf-8

import chainer
import chainer.functions as F


class Continue(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            if i == y:
                continue
            ret = ret + i
        return ret


class ContinueNested(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            for j in range(x):
                if i == y:
                    if j == y:
                        continue
                ret = ret + j
        return ret


class ContinueNestedMultiple(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        for i in range(x):
            if i == y - 1:
                continue
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
    testtools.generate_testcase(Continue, [x, y], subname='continue')
    testtools.generate_testcase(ContinueNested, [x, y], subname='continue_nested')
    testtools.generate_testcase(ContinueNestedMultiple, [x, y], subname='continue_nested_multiple')


if __name__ == '__main__':
    main()
