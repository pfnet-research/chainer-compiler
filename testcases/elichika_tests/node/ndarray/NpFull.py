# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
from chainer_compiler.elichika import testtools
import numpy as np


class Full(chainer.Chain):
    def forward(self):
        y1 = np.full((3, 4), 42)
        return y1


class FullDtype(chainer.Chain):
    def forward(self):
        y1 = np.full((3, 4), 42, dtype=np.float32)
        return y1


# ======================================

def main():
    testtools.generate_testcase(Full, [], subname='none')

    testtools.generate_testcase(FullDtype, [], subname='dtype')


if __name__ == '__main__':
    main()
