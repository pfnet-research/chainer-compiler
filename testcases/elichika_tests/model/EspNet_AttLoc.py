#!/usr/bin/env python
#
# AttLoc from EspNet's e2e_asr.py.
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


# EDIT(hamaji): Use linear_tensor_3d.
def linear_tensor_3d(linear, x):
    '''Apply linear matrix operation only for the last dimension of a tensor

    :param Link linear: Linear link (M x N matrix)
    :param Variable x: Tensor (D_1 x D_2 x M matrix)
    :return:
    :param Variable y: Tensor (D_1 x D_2 x N matrix)
    '''
    return linear(x, n_batch_axes=2)


# location based attention
class AttLoc(chainer.Chain):
    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        with self.init_scope():
            self.mlp_enc = L.Linear(eprojs, att_dim)
            self.mlp_dec = L.Linear(dunits, att_dim, nobias=True)
            self.mlp_att = L.Linear(aconv_chans, att_dim, nobias=True)
            self.loc_conv = L.Convolution2D(1, aconv_chans, ksize=(
                1, 2 * aconv_filts + 1), pad=(0, aconv_filts))
            self.gvec = L.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans

    def reset(self):
        '''reset states

        :return:
        '''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def forward(self, enc_hs, dec_z, att_prev):
        '''AttLoc forward

        :param enc_hs:
        :param dec_z:
        :param att_prev:
        :param scaling:
        :return:
        '''
        # EDIT(hamaji): scaling is now a local variable.
        scaling = 2.0
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor_3d(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z_new = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z_new = F.reshape(dec_z, (batch, self.dunits))

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)

        # TODO(watanabe) use <chainer variable>.reshpae(), instead of F.reshape()
        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(
            F.reshape(att_prev, (batch, 1, 1, self.h_length)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor_3d(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.broadcast_to(
            F.expand_dims(self.mlp_dec(dec_z_new), 1), self.pre_compute_enc_h.shape)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # TODO(watanabe) use batch_matmul
        e = F.squeeze(linear_tensor_3d(self.gvec, F.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)), axis=2)
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w

    def original(self, enc_hs, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param enc_hs:
        :param dec_z:
        :param att_prev:
        :param scaling:
        :return:
        '''
        batch = len(enc_hs)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
            self.h_length = self.enc_h.shape[1]
            # utt x frame x att_dim
            self.pre_compute_enc_h = linear_tensor(self.mlp_enc, self.enc_h)

        if dec_z is None:
            dec_z = chainer.Variable(self.xp.zeros(
                (batch, self.dunits), dtype=np.float32))
        else:
            dec_z = F.reshape(dec_z, (batch, self.dunits))

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [self.xp.full(
                hh.shape[0], 1.0 / hh.shape[0], dtype=np.float32) for hh in enc_hs]
            att_prev = [chainer.Variable(att) for att in att_prev]
            att_prev = F.pad_sequence(att_prev)

        # TODO(watanabe) use <chainer variable>.reshpae(), instead of F.reshape()
        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(
            F.reshape(att_prev, (batch, 1, 1, self.h_length)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = F.swapaxes(F.squeeze(att_conv, axis=2), 1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = linear_tensor(self.mlp_att, att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = F.broadcast_to(
            F.expand_dims(self.mlp_dec(dec_z), 1), self.pre_compute_enc_h.shape)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        # TODO(watanabe) use batch_matmul
        e = F.squeeze(linear_tensor(self.gvec, F.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)), axis=2)
        # Applying a minus-large-number filter to make a probability value zero for a padded area
        # simply degrades the performance, and I gave up this implementation
        # Apply a scaling to make an attention sharp
        w = F.softmax(scaling * e)

        # weighted sum over flames
        # utt x hdim
        c = F.sum(self.enc_h * F.broadcast_to(F.expand_dims(w, 2), self.enc_h.shape), axis=1)

        return c, w


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
    aconv_chans = 7
    aconv_filts = 6

    model_fn = lambda: AttLoc(eprojs, dunits, att_dim, aconv_chans, aconv_filts)
    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, eprojs).astype(dtype=np.float32))

    # Check if our modification is valid.
    model = model_fn()
    expected = model.original(xs, None, None)
    model.reset()
    actual = model.forward(xs, None, None)
    for e, a in zip(expected, actual):
        assert np.allclose(e.array, a.array)

    testtools.generate_testcase(model_fn, [xs, None, None])


if __name__ == '__main__':
    main()
