# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F


class Full(chainer.Chain):
    def forward(self):
        y1 = np.full((3, 4), 42)
        return y1


class FullDtype(chainer.Chain):
    def forward(self):
        y1 = np.full((3, 4), 42, dtype=np.float32)
        return y1


# ======================================

import chainer_compiler.ch2o

if __name__ == '__main__':
    ch2o.generate_testcase(Full, [])

    ch2o.generate_testcase(FullDtype, [], subname='dtype')
