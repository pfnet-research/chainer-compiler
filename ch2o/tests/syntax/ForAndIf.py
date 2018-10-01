# coding: utf-8

import chainer
import chainer.functions as F


class LazyInit(chainer.Chain):
    def forward(self, x):
        y = None
        for i in range(x):
            if y is None:
                y = 42
            y += i
        return y


# ======================================


import ch2o
import numpy as np


if __name__ == '__main__':
    ch2o.generate_testcase(LazyInit(), [5], subname='lazy_init')
