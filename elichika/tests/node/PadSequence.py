# coding: utf-8

import chainer
import chainer.functions as F

# Network definition


class PadSequence(chainer.Chain):
    def forward(self, xs):
        y1 = F.pad_sequence(xs)
        return y1


class PadSequenceLength(chainer.Chain):
    def forward(self, xs):
        y1 = F.pad_sequence(xs, length=20)
        return y1


class PadSequencePadding(chainer.Chain):
    def forward(self, xs):
        y1 = F.pad_sequence(xs, padding=-1)
        return y1


# ======================================

import testtools
import numpy as np

def main():
    import numpy as np
    np.random.seed(314)

    ls = np.random.randint(0, 10, size=5)
    x = [np.random.rand(i).astype(np.float32) for i in ls]
    testtools.generate_testcase(PadSequence, [x], subname='basic')

    testtools.generate_testcase(PadSequenceLength, [x], subname='length')

    testtools.generate_testcase(PadSequencePadding, [x], subname='padding')

if __name__ == '__main__':
    main()
