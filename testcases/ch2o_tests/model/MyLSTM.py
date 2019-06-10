#!/usr/bin/env python
#
# A simple implementation of one-layer NStepLSTM in Python.
#

import argparse
import datetime

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from testcases.ch2o_tests.utils import sequence_utils


class MyLSTM(chainer.Chain):

    def __init__(self, num_hidden, batch_size, sequence_length):
        super(MyLSTM, self).__init__()
        with self.init_scope():
            self.l = L.Linear(num_hidden * 2, num_hidden * 4)
        self.num_hidden = num_hidden
        self.sequence_length = sequence_length
        self.initial_h = np.zeros((batch_size, self.num_hidden),
                                  dtype=np.float32)
        self.initial_c = np.zeros((batch_size, self.num_hidden),
                                  dtype=np.float32)
        self.batch_size = batch_size

    def forward(self, xs, h, c, mask):
        batch_size = len(xs)
        lens = [x.shape[0] for x in xs]
        #max_len = max(lens)
        max_len = self.sequence_length
        #mask = (np.expand_dims(np.arange(max_len), 0) <
        #        np.expand_dims(lens, 1)).astype(np.float)
        #h = np.zeros((batch_size, self.num_hidden), dtype=np.float32)
        #c = np.zeros((batch_size, self.num_hidden), dtype=np.float32)
        #h = self.initial_h
        #c = self.initial_c
        inputs = F.pad_sequence(xs)
        for time in range(max_len):
            x = inputs[:, time]
            input = F.concat((x, h), axis=1)
            gate = self.l(input)
            i = gate[:, 0:self.num_hidden]
            o = gate[:, self.num_hidden:self.num_hidden*2]
            f = gate[:, self.num_hidden*2:self.num_hidden*3]
            nc = gate[:, self.num_hidden*3:self.num_hidden*4]
            #i, o, f, nc = F.split_axis(gate, 4, axis=1)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            f = F.sigmoid(f)
            nc = F.tanh(nc)
            nc = f * c + i * nc
            nh = o * F.tanh(nc)
            m = mask[:, time]
            pmask = F.reshape(m, (self.batch_size,))
            pmask = F.broadcast_to(F.expand_dims(pmask, axis=1),
                                   (self.batch_size, self.num_hidden))
            nmask = 1.0 - pmask
            h = nh * pmask + h * nmask
        return h


# from https://github.com/chainer/chainer/blob/master/examples/seq2seq/seq2seq.py

from chainer_compiler import ch2o


# TODO(hamaji): This is broken. Fix it.
def run_with_n_step_lstm(xs, h, c, w, b):
    xs = F.transpose_sequence(xs)
    print(w.shape)
    wx, wh = F.split_axis(w, 2, 1)
    ws = F.split_axis(wx, 4, 0) + F.split_axis(wh, 4, 0)
    b = b / 2
    bs = F.split_axis(b, 4, 0) * 2
    print(bs)
    h, _, _ = F.n_step_lstm(1, 0.0, h, c, ws, bs, xs)
    return h


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    batch_size = 3
    sequence_length = 4
    num_vocabs = 10
    num_hidden = 5

    model_fn = lambda: MyLSTM(num_hidden, batch_size, sequence_length)

    labels, lengths = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in lengths:
        xs.append(np.random.rand(l, num_hidden).astype(dtype=np.float32))

    h = np.zeros((batch_size, num_hidden), dtype=np.float32)
    c = np.zeros((batch_size, num_hidden), dtype=np.float32)
    mask = (np.expand_dims(np.arange(sequence_length), 0) <
            np.expand_dims(lengths, 1)).astype(np.float32)

    args = [xs, h, c, mask]

    #print(model_fn()(*args))
    #print(run_with_n_step_lstm(xs, h, c, model.l.W, model.l.b))

    ch2o.generate_testcase(model_fn, args)
