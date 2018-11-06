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


# TODO(watanabe) explanation of VGG2L, VGG2B (Block) might be better
class VGG2L(chainer.Chain):
    def __init__(self, in_channel=1, nobias=False):
        super(VGG2L, self).__init__()
        with self.init_scope():
            # CNN layer (VGG motivated)
            # EDIT(hamaji): Add `nobias=True`.
            # TODO(hamaji): Check if why gradient of bias is large.
            self.conv1_1 = L.Convolution2D(in_channel, 64, 3, stride=1, pad=1,
                                           nobias=nobias)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)

        self.in_channel = in_channel

    def forward(self, xs, ilens):
        '''VGG2L forward

        :param xs:
        :param ilens:
        :return:
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        xs = F.pad_sequence(xs)

        # x: utt x 1 (input channel num) x frame x dim
        xs = F.swapaxes(F.reshape(
            xs, (xs.shape[0], xs.shape[1], self.in_channel, xs.shape[2] // self.in_channel)), 1, 2)

        xs = F.relu(self.conv1_1(xs))
        xs = F.relu(self.conv1_2(xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        xs = F.relu(self.conv2_1(xs))
        xs = F.relu(self.conv2_2(xs))
        xs = F.max_pooling_2d(xs, 2, stride=2)

        # change ilens accordingly
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)
        ilens = self.xp.array(self.xp.ceil(self.xp.array(
            ilens, dtype=np.float32) / 2), dtype=np.int32)

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = F.swapaxes(xs, 1, 2)
        xs = F.reshape(
            xs, (xs.shape[0], xs.shape[1], xs.shape[2] * xs.shape[3]))
        xs = [xs[i, :ilens[i], :] for i in range(len(ilens))]

        return xs, ilens


class VGG2LBackprop(chainer.Chain):
    def __init__(self, idim):
        super(VGG2LBackprop, self).__init__()
        with self.init_scope():
            self.vgg = VGG2L(idim, nobias=True)

    def forward(self, xs, ilens):
        xs, ilens = self.vgg(xs, ilens)
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

    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, sequence_length, num_vocabs)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, idim).astype(dtype=np.float32))

    ch2o.generate_testcase(lambda: VGG2L(1), [xs, ilens])

    ch2o.generate_testcase(lambda:  VGG2LBackprop(1),
                           [xs, ilens], backprop=True)
