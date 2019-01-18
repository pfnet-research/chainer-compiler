#!/usr/bin/env python

import numpy

import chainer
import chainer.functions as F
import chainer.links as L

#from ch2o.test_args import dprint


class LinkInFor(chainer.Chain):

    def __init__(self, num_hidden):
        super(LinkInFor, self).__init__()
        with self.init_scope():
            self.l = L.Linear(num_hidden, num_hidden)

    def forward(self, x, h, indices):
        for i in indices:
            h = h + self.l(x[:, i])
        return h


import testtools

if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    batch_size = 3
    num_hidden = 5
    sequence_length = 4

    model = LinkInFor(num_hidden)

    x = np.random.rand(
        batch_size, sequence_length, num_hidden).astype(np.float32)
    h = np.random.rand(batch_size, num_hidden).astype(np.float32)

    args = [x, h, np.arange(sequence_length)]
    #dprint(model(*args))
    testtools.generate_testcase(model, args)
