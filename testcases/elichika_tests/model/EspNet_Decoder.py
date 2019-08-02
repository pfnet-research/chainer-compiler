#!/usr/bin/env python
#
# Decoder from EspNet's e2e_asr.py.
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

from testcases.elichika_tests.utils import sequence_utils
from testcases.elichika_tests.model.EspNet_AttDot import AttDot
from testcases.elichika_tests.model.EspNet_AttLoc import AttLoc
from testcases.elichika_tests.model.StatelessLSTM import StatelessLSTM


def _mean(xs):
    sum_len = 0
    for x in xs:
        sum_len += len(x)
    return sum_len / len(xs)


def _flatten(xs):
    return F.reshape(xs, (xs.size,))


class Decoder(chainer.Chain):
    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att_dim,
                 aconv_chans=None, aconv_filts=None,
                 verbose=0, char_list=None, labeldist=None,
                 lsm_weight=0., sampling_probability=0.0,
                 use_chainer=False):
        super(Decoder, self).__init__()
        lstm_cls = L.StatelessLSTM if use_chainer else StatelessLSTM
        with self.init_scope():
            # EDIT(hamaji): Use L.Embed instead of DL.EmbedID.
            # self.embed = DL.EmbedID(odim, dunits)
            self.embed = L.EmbedID(odim, dunits)
            # EDIT(hamaji): Use StatelessLSTM instead of Chainer's.
            # self.lstm0 = L.StatelessLSTM(dunits + eprojs, dunits)
            self.lstm0 = lstm_cls(dunits + eprojs, dunits)
            # EDIT(hamaji): Limit the number of decoder layers.
            # for l in six.moves.range(1, dlayers):
            assert dlayers <= 2
            for l in six.moves.range(1, 2):
                # EDIT(hamaji): Use StatelessLSTM instead of Chainer's.
                # setattr(self, 'lstm%d' % l, L.StatelessLSTM(dunits, dunits))
                setattr(self, 'lstm%d' % l, lstm_cls(dunits, dunits))
            self.output = L.Linear(dunits, odim)
            # EDIT(hamaji): attention parameters are passed instead of `att`.
            if aconv_chans is None:
                self.att = AttDot(eprojs, dunits, att_dim)
            else:
                self.att = AttLoc(eprojs, dunits, att_dim,
                                  aconv_chans, aconv_filts)

        self.loss = None
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
        eys = self.embed(pad_ys_in)
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
            # EDIT(hamaji): Unrolled, etc.
            c_list_new = []
            z_list_new = []
            c_new, z_new = self.lstm0(c_list[0], z_list[0], ey)
            c_list_new.append(c_new)
            z_list_new.append(z_new)
            if self.dlayers > 1:
                c_new, z_new = self.lstm1(c_list[1], z_list[1], z_list_new[-1])
                c_list_new.append(c_new)
                z_list_new.append(z_new)
            # for l in six.moves.range(1, self.dlayers):
            #     c_new, z_new = self['lstm%d' % l](c_list[l], z_list[l], z_list_new[-1])
            #     c_list_new.append(c_new)
            #     z_list_new.append(z_new)
            c_list = c_list_new
            z_list = z_list_new
            z_all.append(z_list[-1])

        z_all = F.reshape(F.stack(z_all, axis=1),
                          (batch * olength, self.dunits))
        # compute loss
        y_all = self.output(z_all)
        # EDIT(hamaji): `np.flatten` implemented by ourselves.
        # self.loss = F.softmax_cross_entropy(y_all, F.flatten(pad_ys_out))
        self.loss = F.softmax_cross_entropy(y_all, _flatten(pad_ys_out))
        # -1: eos, which is removed in the loss computation
        # EDIT(hamaji): `np.mean` implemented by a naive loop.
        # self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        self.loss *= _mean(ys_in) - 1
        # EDIT(hamaji): No need to compute accuracy.
        # acc = F.accuracy(y_all, F.flatten(pad_ys_out), ignore_label=-1)
        # logging.info('att loss:' + str(self.loss.data))

        # EDIT(hamaji): Skip verbose logging.
        # # show predicted character sequence for debug
        # if self.verbose > 0 and self.char_list is not None:
        #     y_hat = F.reshape(y_all, (batch, olength, -1))
        #     y_true = pad_ys_out
        #     for (i, y_hat_), y_true_ in zip(enumerate(y_hat.data), y_true.data):
        #         if i == MAX_DECODER_OUTPUT:
        #             break
        #         idx_hat = self.xp.argmax(y_hat_[y_true_ != -1], axis=1)
        #         idx_true = y_true_[y_true_ != -1]
        #         seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
        #         seq_true = [self.char_list[int(idx)] for idx in idx_true]
        #         seq_hat = "".join(seq_hat).replace('<space>', ' ')
        #         seq_true = "".join(seq_true).replace('<space>', ' ')
        #         logging.info("groundtruth[%d]: " % i + seq_true)
        #         logging.info("prediction [%d]: " % i + seq_hat)

        # EDIT(hamaji): Skip `labeldist` thing.
        # if self.labeldist is not None:
        #     if self.vlabeldist is None:
        #         self.vlabeldist = chainer.Variable(self.xp.asarray(self.labeldist))
        #     loss_reg = - F.sum(F.scale(F.log_softmax(y_all), self.vlabeldist, axis=1)) / len(ys_in)
        #     self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        # EDIT(hamaji): Return loss only.
        # return self.loss, acc
        return self.loss

    def original(self, hs, ys):
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
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = F.argmax(F.log_softmax(z_out), axis=1)
                z_out = self.embed(z_out)
                ey = F.hstack((z_out, att_c))  # utt x (zdim + hdim)
            else:
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


from chainer_compiler.elichika import testtools


def main():
    import numpy as np
    np.random.seed(43)

    eprojs = 3
    dunits = 4
    att_dim = 5
    batch_size = 3
    sequence_length = 4
    num_vocabs = 10
    dlayers = 2
    odim = 11
    sos = odim - 1
    eos = odim - 1
    aconv_chans = 7
    aconv_filts = 6

    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    hs = []
    for l in ilens:
        hs.append(np.random.rand(l, eprojs).astype(dtype=np.float32))

    ys, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, odim)

    def gen_test(model_fn, subname=None):
        model = model_fn()
        # Check if our modification is valid.
        expected, _ = model.original(hs, ys)
        actual = model.forward(hs, ys)
        assert np.allclose(expected.array, actual.array)

        testtools.generate_testcase(model_fn, [hs, ys], subname=subname)

    def model_fn():
        # att = AttDot(eprojs, dunits, att_dim)
        # dec = Decoder(eprojs, odim, dlayers, dunits, sos, eos, att)
        dec = Decoder(eprojs, odim, dlayers, dunits, sos, eos, att_dim)
        return dec

    gen_test(model_fn)

    testtools.generate_testcase(model_fn, [hs, ys], backprop=True)

    def model_fn():
        dec = Decoder(eprojs, odim, dlayers, dunits, sos, eos,
                      att_dim, aconv_chans, aconv_filts)
        return dec

    gen_test(model_fn, subname='attloc')

    testtools.generate_testcase(model_fn, [hs, ys], subname='attloc', backprop=True)


if __name__ == '__main__':
    main()
