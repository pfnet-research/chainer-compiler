# coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import numpy as np
import testtools


class A(chainer.Chain):

    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        return chainer.Variable(x)


# ======================================

def main():
    testtools.generate_testcase(A(), [np.array(42)])

if __name__ == '__main__':
    main()