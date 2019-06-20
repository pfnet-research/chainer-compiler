# coding: utf-8

import chainer
import chainer.functions as F


class BoolOp(chainer.Chain):
    def forward(self, x, y):
        test1 = x and y
        test2 = x or y
        test3 = y and x or True
        test4 = y and x and False or True
        ret = test1 or test2 and test3 or test4
        return ret

class BoolOpIf(chainer.Chain):
    def forward(self, x, y):
        ret = 0
        if x and y:
            ret += 1
        if x or y:
            ret += 2
        if y and x or True:
            ret += 3
        if y and x and False or True:
            ret += 4
        return x or y


# ======================================


from chainer_compiler.elichika import testtools
import numpy as np
from itertools import product

def main():
    for i, (x, y) in enumerate(product([True, False], repeat=2)):
        testtools.generate_testcase(BoolOp, [x, y], subname='boolop%d' % i)
        testtools.generate_testcase(BoolOpIf, [x, y], subname='boolop_if%d' % i)


if __name__ == '__main__':
    main()
