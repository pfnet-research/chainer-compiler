#!/usr/bin/env python
#
# AttDot from EspNet's e2e_asr.py.
#

import argparse
import datetime
import logging
import random
import six

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from tests.utils import sequence_utils


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

    def precompute(self, enc_hs, dec_z, att_prev, scaling=2.0):
        assert self.pre_compute_enc_h is None
        self.enc_h = F.pad_sequence(enc_hs)  # utt x frame x hdim
        self.h_length = self.enc_h.shape[1]
        # utt x frame x att_dim
        self.pre_compute_enc_h = F.tanh(
            linear_tensor(self.mlp_enc, self.enc_h))

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


class Decoder(chainer.Chain):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0):
        super(Decoder, self).__init__()
        with self.init_scope():
            # EDIT(hamaji): Use L.Embed instead of DL.EmbedID.
            # self.embed = DL.EmbedID(odim, dunits)
            self.embed = L.EmbedID(odim, dunits)
            self.lstm0 = L.StatelessLSTM(dunits + eprojs, dunits)
            for l in six.moves.range(1, dlayers):
                setattr(self, 'lstm%d' % l, L.StatelessLSTM(dunits, dunits))
            self.output = L.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dlayers = dlayers
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability

    def forward(self, hs, ys):
        '''Decoder forward

        :param Variable hs:
        :param Variable ys:
        :return:
        '''
        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = self.xp.array([self.eos], 'i')
        sos = self.xp.array([self.sos], 'i')
        ys_in = [F.concat([sos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        pad_ys_in = F.pad_sequence(ys_in, padding=self.eos)
        pad_ys_out = F.pad_sequence(ys_out, padding=-1)

        # get dim, length info
        batch = pad_ys_out.shape[0]
        olength = pad_ys_out.shape[1]
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(self.xp.array([h.shape[0] for h in hs])))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str(self.xp.array([y.shape[0] for y in ys_out])))

        # initialization
        c_list = [None]  # list of cell state of each layer
        z_list = [None]  # list of hidden state of each layer
        for l in six.moves.range(1, self.dlayers):
            c_list.append(None)
            z_list.append(None)
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.embed(pad_ys_in)  # utt x olen x zdim
        eys = F.separate(eys, axis=1)

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs, z_list[0], att_w)
            # EDIT(hamaji): No scheduled sampling.
            # if i > 0 and random.random() < self.sampling_probability:
            #     logging.info(' scheduled sampling ')
            #     z_out = self.output(z_all[-1])
            #     z_out = F.argmax(F.log_softmax(z_out), axis=1)
            #     z_out = self.embed(z_out)
            #     ey = F.hstack((z_out, att_c))  # utt x (zdim + hdim)
            # else:
            #     ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            ey = F.hstack((eys[i], att_c))  # utt x (zdim + hdim)
            c_list[0], z_list[0] = self.lstm0(c_list[0], z_list[0], ey)
            for l in six.moves.range(1, self.dlayers):
                c_list[l], z_list[l] = self['lstm%d' % l](c_list[l], z_list[l], z_list[l - 1])
            z_all.append(z_list[-1])

        z_all = F.reshape(F.stack(z_all, axis=1),
                          (batch * olength, self.dunits))
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.softmax_cross_entropy(y_all, F.flatten(pad_ys_out))
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = F.accuracy(y_all, F.flatten(pad_ys_out), ignore_label=-1)
        logging.info('att loss:' + str(self.loss.data))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            y_hat = F.reshape(y_all, (batch, olength, -1))
            y_true = pad_ys_out
            for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
                idx_true = y_true_[y_true_ != -1]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat).replace('<space>', ' ')
                seq_true = "".join(seq_true).replace('<space>', ' ')
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = chainer.Variable(self.xp.asarray(self.labeldist))
            loss_reg = - F.sum(F.scale(F.log_softmax(y_all), self.vlabeldist, axis=1)) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc


import ch2o


if __name__ == '__main__':
    import numpy as np
    np.random.seed(314)

    eprojs = 3
    dunits = 4
    att_dim = 5
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10
    dlayers = 3
    odim = 11
    sos = odim - 1
    eos = odim - 2

    def model_fn():
        att = AttDot(eprojs, dunits, att_dim)
        dec = Decoder(eprojs, odim, dlayers, dunits, sos, eos, att)
        return dec

    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    hs = []
    for l in ilens:
        hs.append(np.random.rand(l, eprojs).astype(dtype=np.float32))

    ys, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, odim)

    # Check if our modification is valid.
    #expected = model_fn().original(hs, ys)
    actual = model_fn().forward(hs, ys)
    #for e, a in zip(expected, actual):
    #    assert np.allclose(e.array, a.array)

    ch2o.generate_testcase(model_fn, [hs, ys])
