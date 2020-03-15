#!/usr/bin/env python
#
# AttDot from EspNet's e2e_asr.py.
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

from testcases.elichika_tests.utils import sequence_utils


# TODO(kan-bayashi): no need to use linear tensor
def linear_tensor(linear, x):
    '''Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x ... x M matrix)
    :return:
    :param Variable y: Tensor (D_1 x D_2 x ... x N matrix)
    '''
    y = linear(F.reshape(x, (-1, x.shape[-1])))
    return F.reshape(y, (x.shape[:-1] + (-1,)))


class AttDot(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        '''reset states

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs, dec_z, att_prev):
        '''AttDot forward

        :param enc_hs:
        :param dec_z:
        :param scaling:
        :return:
        '''
        # EDIT(hamaji): scaling is now a local variable.
        scaling = 2.0
        batch = len(enc_hs)

        # EDIT(momohatt): Make sure to initialize self.enc_h
        if self.enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim

        if self.pre_compute_enc_h is None:
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = F.tanh(
                linear_tensor(self.mlp_enc, self.enc_h))

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # <phi (h_t), psi (s)> for all t
        u = F.broadcast_to(F.expand_dims(F.tanh(self.mlp_dec(dec_z)), 1),
                           self.pre_compute_enc_h.shape)
        e = F.sum(self.pre_compute_enc_h * u, axis=2)  # utt x frame
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)
        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w

    def original(self, enc_hs, dec_z, att_prev, scaling=2.0):
        '''AttDot forward

        :param enc_hs:
        :param dec_z:
        :param scaling:
        :return:
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = F.tanh(
                linear_tensor(self.mlp_enc, self.enc_h))

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # <phi (h_t), psi (s)> for all t
        u = F.broadcast_to(F.expand_dims(F.tanh(self.mlp_dec(dec_z)), 1),
                           self.pre_compute_enc_h.shape)
        e = F.sum(self.pre_compute_enc_h * u, axis=2)  # utt x frame
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)
        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


class AttDotBackprop(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim):
        super(AttDotBackprop, self).__init__()
        with self.init_scope():
            self.l = AttDot(eprojs, dunits, att_dim)

    def forward(self, enc_hs, dec_z, att_prev):
        c, w = self.l(enc_hs, dec_z, att_prev)
        return F.matmul(c, w)


from chainer_compiler.elichika import testtools


def main():
    import numpy as np
    np.random.seed(314)

    eprojs = 3
    dunits = 4
    att_dim = 5
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10

    model_fn = lambda: AttDot(eprojs, dunits, att_dim)
    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, eprojs).astype(np.float32))

    # Check if our modification is valid.
    expected = model_fn().original(xs, None, None)
    actual = model_fn().forward(xs, None, None)
    for e, a in zip(expected, actual):
        assert np.allclose(e.array, a.array)

    testtools.generate_testcase(model_fn, [xs, None, None])

    z = np.random.rand(batch_size, dunits).astype(np.float32)
    testtools.generate_testcase(lambda: AttDotBackprop(eprojs, dunits, att_dim),
                           [xs, z, None],
                           backprop=True)


if __name__ == '__main__':
    main()
