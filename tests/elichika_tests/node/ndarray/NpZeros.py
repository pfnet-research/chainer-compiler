# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import testtools
import numpy as np


class A(chainer.Chain):
    def forward(self):
        y1 = np.zeros((3, 4), dtype=np.float32)
        return y1


# ======================================

def main():
    testtools.generate_testcase(A(), [])


if __name__ == '__main__':
    main()
