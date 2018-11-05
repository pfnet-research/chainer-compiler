#!/usr/bin/env python
#
# BLSTM from EspNet's e2e_asr.py.
#

import argparse
import datetime
import logging

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from tests.utils import sequence_utils


class BLSTM(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        with self.init_scope():
            self.nblstm = L.NStepBiLSTM(elayers, idim, cdim, dropout)
            self.l_last = L.Linear(cdim * 2, hdim)

    def forward(self, xs, ilens):
        '''BLSTM forward (the modified version)

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # need to move ilens to cpu
        ilens = cuda.to_cpu(ilens)
        hy, cy, ys = self.nblstm(None, None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # EDIT(hamaji): Unnecessary, as `force_tuple` is True by default.
        # # 1 utterance case, it becomes an array, so need to make a utt tuple
        # if not isinstance(xs, tuple):
        #     xs = [xs]

        return xs, ilens  # x: utt list of frame x dim

    def original(self, xs, ilens):
        '''BLSTM forward (the original implementation)

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        # need to move ilens to cpu
        ilens = cuda.to_cpu(ilens)
        hy, cy, ys = self.nblstm(None, None, xs)
        ys = self.l_last(F.vstack(ys))  # (sum _utt frame_utt) x dim
        xs = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        del hy, cy

        # final tanh operation
        xs = F.split_axis(F.tanh(F.vstack(xs)), np.cumsum(ilens[:-1]), axis=0)

        # 1 utterance case, it becomes an array, so need to make a utt tuple
        if not isinstance(xs, tuple):
            xs = [xs]

        return xs, ilens  # x: utt list of frame x dim


class BLSTMBackprop(chainer.Chain):
    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTMBackprop, self).__init__()
        with self.init_scope():
            self.blstm = BLSTM(idim, elayers, cdim, hdim, dropout)

    def forward(self, xs, ilens):
        xs, ilens = self.blstm(xs, ilens)
        return F.pad_sequence(xs)


import ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    idim = 5
    elayers = 2
    cdim = 3
    hdim = 7
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10

    model = BLSTM(idim, elayers, cdim, hdim, 0)
    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, idim).astype(dtype=np.float32))

    # Check if our modification is valid.
    expected = model.original(xs, ilens)
    actual = model.forward(xs, ilens)
    for e, a in zip(expected[0], actual[0]):
        assert np.allclose(e.array, a.array)
    assert np.allclose(expected[1], actual[1])

    ch2o.generate_testcase(model, [xs, ilens])

    ch2o.generate_testcase(BLSTMBackprop(idim, elayers, cdim, hdim, 0),
                           [xs, ilens], backprop=True)
