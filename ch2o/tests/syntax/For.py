# coding: utf-8

import chainer
import chainer.functions as F


def _gen_random_sequence(batch_size, sequence_length, num_vocabs):
    lengths = np.random.randint(2, sequence_length, size=batch_size)
    lengths = np.flip(np.sort(lengths), axis=0)
    # At least a single element should have the maximum sequence
    # length to avoid a shape mismatch.
    lengths[0] = sequence_length
    labels = np.random.randint(
        2, num_vocabs, size=(batch_size, sequence_length))
    return labels, lengths


class A(chainer.Chain):
    def forward(self, xs, p):
        v = []
        for i in range(p):
            v.append(xs[:i])
        return v



class B(chainer.Chain):
    def forward(self, xs, l):
        inputs = F.pad_sequence(xs)
        h = inputs[:, 0]
        for time in range(l):
            h = inputs[:, time]
        return h


# ======================================


import ch2o
import numpy as np


if __name__ == '__main__':
    model = A()

    v = np.random.rand(10).astype(np.float32)
    p = np.int64(5)
    ch2o.generate_testcase(model, [v, p])

    model = B()
    length = 4
    xs = []
    for i in range(length):
        xs.append(np.random.rand(length, 5).astype(dtype=np.float32))
    args = [xs, length]
    ch2o.generate_testcase(model, args, subname='closure_bug')
