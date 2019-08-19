#!/usr/bin/env python
#
# E2E from EspNet's e2e_asr.py.
#

import argparse
import datetime
import logging
import random
import six
import sys
import time

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import function_hooks
from chainer import training
from chainer.training import extensions

from testcases.elichika_tests.utils import sequence_utils
from testcases.elichika_tests.model.EspNet_AttDot import AttDot
from testcases.elichika_tests.model.EspNet_AttLoc import AttLoc
from testcases.elichika_tests.model.EspNet_BLSTM import BLSTM
from testcases.elichika_tests.model.EspNet_VGG2L import VGG2L
from testcases.elichika_tests.model.EspNet_Decoder import Decoder


def get_vgg2l_odim(idim, in_channel=3, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


class Encoder(chainer.Chain):
    '''ENCODER NETWORK CLASS

    This is the example of docstring.

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param str subsample: subsampling number e.g. 1_2_2_2_1
    :param float dropout: dropout rate
    :return:

    '''

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1, nobias=False):
        super(Encoder, self).__init__()
        with self.init_scope():
            if etype == 'blstm':
                self.enc1 = BLSTM(idim, elayers, eunits, eprojs, dropout)
                logging.info('BLSTM without projection for encoder')
            elif etype == 'blstmp':
                self.enc1 = BLSTMP(idim, elayers, eunits,
                                   eprojs, subsample, dropout)
                logging.info('BLSTM with every-layer projection for encoder')
            elif etype == 'vggblstmp':
                self.enc1 = VGG2L(in_channel, nobias=nobias)
                self.enc2 = BLSTMP(get_vgg2l_odim(
                    idim, in_channel=in_channel), elayers, eunits, eprojs, subsample, dropout)
                logging.info('Use CNN-VGG + BLSTMP for encoder')
            elif etype == 'vggblstm':
                self.enc1 = VGG2L(in_channel, nobias=nobias)
                self.enc2 = BLSTM(get_vgg2l_odim(
                    idim, in_channel=in_channel), elayers, eunits, eprojs, dropout)
                logging.info('Use CNN-VGG + BLSTM for encoder')
            else:
                logging.error(
                    "Error: need to specify an appropriate encoder archtecture")
                sys.exit()

        self.etype = etype

    def forward(self, xs, ilens):
        '''Encoder forward

        :param xs:
        :param ilens:
        :return:
        '''
        # EDIT(hamaji): Always use VGG2L.
        xs, ilens = self.enc1(xs, ilens)
        xs, ilens = self.enc2(xs, ilens)
        return xs, ilens

        # Edit(rchouras): Commenting this out as it is No-op because of above return.
        # if self.etype == 'blstm':
        #     xs, ilens = self.enc1(xs, ilens)
        # elif self.etype == 'blstmp':
        #     xs, ilens = self.enc1(xs, ilens)
        # elif self.etype == 'vggblstmp':
        #     xs, ilens = self.enc1(xs, ilens)
        #     xs, ilens = self.enc2(xs, ilens)
        # elif self.etype == 'vggblstm':
        #     xs, ilens = self.enc1(xs, ilens)
        #     xs, ilens = self.enc2(xs, ilens)
        # else:
        #     logging.error(
        #         "Error: need to specify an appropriate encoder archtecture")
        #     sys.exit()

        # return xs, ilens


class E2E(chainer.Chain):
    def __init__(self, idim, odim, args, nobias=False, use_chainer=False):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype == 'blstmp':
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            # EDIT(hamaji): Remove the warning.
            # logging.warning(
            #     'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
            pass
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        with self.init_scope():
            # encoder
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                               self.subsample, args.dropout_rate, 1, nobias=nobias)
            # ctc
            ctc_type = vars(args).get("ctc_type", "chainer")
            if ctc_type == 'chainer':
                logging.info("Using chainer CTC implementation")
                self.ctc = CTC(odim, args.eprojs, args.dropout_rate)
            elif ctc_type == 'warpctc':
                logging.info("Using warpctc CTC implementation")
                self.ctc = WarpCTC(odim, args.eprojs, args.dropout_rate)
            # attention
            # EDIT(hamaji): Decoder's interface was changed.
            # if args.atype == 'dot':
            #     self.att = AttDot(args.eprojs, args.dunits, args.adim)
            # elif args.atype == 'location':
            #     self.att = AttLoc(args.eprojs, args.dunits,
            #                       args.adim, args.aconv_chans, args.aconv_filts)
            # elif args.atype == 'noatt':
            #     self.att = NoAtt()
            # else:
            #     logging.error(
            #         "Error: need to specify an appropriate attention archtecture")
            #     sys.exit()
            # decoder
            # EDIT(hamaji): Decoder's interface was changed.
            # self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
            #                    self.sos, self.eos, self.att, self.verbose, self.char_list,
            #                    labeldist, args.lsm_weight, args.sampling_probability)
            if args.atype == 'dot':
                self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                                   self.sos, self.eos, args.adim,
                                   use_chainer=use_chainer)
            elif args.atype == 'location':
                self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                                   self.sos, self.eos, args.adim,
                                   args.aconv_chans, args.aconv_filts,
                                   use_chainer=use_chainer)
            else:
                raise RuntimeError('Not supported: %s' % args.atype)

    def forward(self, xs, ilens, ys):
        '''E2E forward

        :param data:
        :return:
        '''
        # 1. encoder
        hs, ilens = self.enc(xs, ilens)

        # 3. CTC loss
        # EDIT(hamaji): Skip CTC.
        # if self.mtlalpha == 0:
        #     loss_ctc = None
        # else:
        #     loss_ctc = self.ctc(hs, ys)
        loss_ctc = None

        # EDIT(hamaji): Decoder only returns loss.
        return self.dec(hs, ys)

        # Edit(rchouras): Commenting this out as it is No-op because of above return.
        # 4. attention loss
        # if self.mtlalpha == 1:
        #     loss_att = None
        #     acc = None
        # else:
        #     loss_att, acc = self.dec(hs, ys)

        # return loss_ctc, loss_att, acc

    def original(self, xs, ilens, ys):
        '''E2E forward

        :param data:
        :return:
        '''
        # 1. encoder
        hs, ilens = self.enc(xs, ilens)

        # 3. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs, ys)

        # EDIT(hamaji): Decoder only returns loss.
        return self.dec(hs, ys)

        # 4. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs, ys)

        return loss_ctc, loss_att, acc


from chainer_compiler.elichika import testtools


class Args(object):
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def gen_inputs(batch_size, ilen, idim, olen, odim):
    labels, ilens = sequence_utils.gen_random_sequence(
        batch_size, ilen, idim)
    xs = []
    for l in ilens:
        xs.append(np.random.rand(l, idim).astype(dtype=np.float32))

    ys, _ = sequence_utils.gen_random_sequence(
        batch_size, olen, odim)
    return xs, ilens, ys


def test_recipe():
    aconv_chans = 7
    aconv_filts = 6
    att_dim = 5
    batch_size = 3
    dlayers = 2
    dunits = 4
    elayers = 2
    eprojs = 3
    eunits = 3
    idim = 6
    odim = 11
    ilen = 4
    olen = 4

    args = Args({
        'aconv_chans': aconv_chans,
        'aconv_filts': aconv_filts,
        'adim': att_dim,
        #'atype': 'dot',
        'atype': 'location',
        'char_list': None,
        'ctc_type': None,
        'dlayers': dlayers,
        'dropout_rate': 0,
        'dunits': dunits,
        'elayers': elayers,
        'eprojs': eprojs,
        'etype': 'vggblstm',
        'eunits': eunits,
        'lsm_type': None,
        'lsm_weight': None,
        'mtlalpha': 0,
        'outdir': None,
        'sampling_probability': None,
        'subsample': None,
        'train_json': None,
        'verbose': False,
    })

    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def csj_recipe():
    aconv_chans = 10
    aconv_filts = 100
    att_dim = 320
    batch_size = 32
    dlayers = 1
    dunits = 1024
    elayers = 4
    eprojs = 320
    eunits = 1024

    # TODO(hamaji): Figure out sane values.
    #ilen = 20
    #olen = 40
    #idim = 10
    #odim = 10
    ilen = 5
    olen = 4
    idim = 10
    odim = 10

    args = Args({
        'aconv_chans': aconv_chans,
        'aconv_filts': aconv_filts,
        'adim': att_dim,
        #'atype': 'dot',
        'atype': 'location',
        'char_list': None,
        'ctc_type': None,
        'dlayers': dlayers,
        'dropout_rate': 0,
        'dunits': dunits,
        'elayers': elayers,
        'eprojs': eprojs,
        'etype': 'vggblstm',
        'eunits': eunits,
        'lsm_type': None,
        'lsm_weight': None,
        'mtlalpha': 0,
        'outdir': None,
        'sampling_probability': None,
        'subsample': None,
        'train_json': None,
        'verbose': False,
    })

    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def csj_small_recipe():
    aconv_chans = 10
    aconv_filts = 100
    att_dim = 320
    batch_size = 32
    dlayers = 2
    dunits = 768
    elayers = 4
    eprojs = 320
    eunits = 768

    ilen = 30
    olen = 10
    idim = 65
    odim = 70

    args = Args({
        'aconv_chans': aconv_chans,
        'aconv_filts': aconv_filts,
        'adim': att_dim,
        #'atype': 'dot',
        'atype': 'location',
        'char_list': None,
        'ctc_type': None,
        'dlayers': dlayers,
        'dropout_rate': 0,
        'dunits': dunits,
        'elayers': elayers,
        'eprojs': eprojs,
        'etype': 'vggblstm',
        'eunits': eunits,
        'lsm_type': None,
        'lsm_weight': None,
        'mtlalpha': 0,
        'outdir': None,
        'sampling_probability': None,
        'subsample': None,
        'train_json': None,
        'verbose': False,
    })

    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def csj_medium_recipe():
    aconv_chans = 10
    aconv_filts = 100
    att_dim = 320
    batch_size = 32
    dlayers = 2
    dunits = 1024
    elayers = 4
    eprojs = 320
    eunits = 1024

    ilen = 30
    olen = 10
    idim = 100
    odim = 100

    args = Args({
        'aconv_chans': aconv_chans,
        'aconv_filts': aconv_filts,
        'adim': att_dim,
        #'atype': 'dot',
        'atype': 'location',
        'char_list': None,
        'ctc_type': None,
        'dlayers': dlayers,
        'dropout_rate': 0,
        'dunits': dunits,
        'elayers': elayers,
        'eprojs': eprojs,
        'etype': 'vggblstm',
        'eunits': eunits,
        'lsm_type': None,
        'lsm_weight': None,
        'mtlalpha': 0,
        'outdir': None,
        'sampling_probability': None,
        'subsample': None,
        'train_json': None,
        'verbose': False,
    })

    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def librispeech_args():
    aconv_chans = 10
    aconv_filts = 100
    att_dim = 1024
    dlayers = 2
    dunits = 1024
    elayers = 5
    eprojs = 1024
    eunits = 1024

    args = Args({
        'aconv_chans': aconv_chans,
        'aconv_filts': aconv_filts,
        'adim': att_dim,
        #'atype': 'dot',
        'atype': 'location',
        'char_list': None,
        'ctc_type': None,
        'dlayers': dlayers,
        'dropout_rate': 0,
        'dunits': dunits,
        'elayers': elayers,
        'eprojs': eprojs,
        'etype': 'vggblstm',
        'eunits': eunits,
        'lsm_type': None,
        'lsm_weight': None,
        'mtlalpha': 0,
        'outdir': None,
        'sampling_probability': None,
        'subsample': None,
        'train_json': None,
        'verbose': False,
    })
    return args


def librispeech_small_recipe():
    batch_size = 3
    ilen = 1500
    olen = 50
    idim = 83
    odim = 5002
    args = librispeech_args()
    args.att_dim = 512
    args.dunits = 512
    args.eprojs = 512
    args.eunits = 512
    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def librispeech_recipe():
    batch_size = 20
    ilen = 1500
    olen = 50
    idim = 83
    odim = 5002
    args = librispeech_args()
    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def cpu_bench_recipe():
    # An extremely
    aconv_chans = 4
    aconv_filts = 4
    att_dim = 4
    batch_size = 4
    dlayers = 2
    dunits = 4
    elayers = 4
    eprojs = 4
    eunits = 4

    ilen = 10
    olen = 40
    idim = 5
    odim = 5

    args = Args({
        'aconv_chans': aconv_chans,
        'aconv_filts': aconv_filts,
        'adim': att_dim,
        #'atype': 'dot',
        'atype': 'location',
        'char_list': None,
        'ctc_type': None,
        'dlayers': dlayers,
        'dropout_rate': 0,
        'dunits': dunits,
        'elayers': elayers,
        'eprojs': eprojs,
        'etype': 'vggblstm',
        'eunits': eunits,
        'lsm_type': None,
        'lsm_weight': None,
        'mtlalpha': 0,
        'outdir': None,
        'sampling_probability': None,
        'subsample': None,
        'train_json': None,
        'verbose': False,
    })

    xs, ilens, ys = gen_inputs(batch_size, ilen, idim, olen, odim)

    return (idim, odim, args), (xs, ilens, ys)


def gen_test():
    (idim, odim, args), (xs, ilens, ys) = test_recipe()

    def gen_test(model_fn, subname=None):
        model = model_fn()
        # Check if our modification is valid.
        expected = model.original(xs, ilens, ys)
        actual = model.forward(xs, ilens, ys)
        assert np.allclose(expected.array, actual.array)

        testtools.generate_testcase(model_fn, [xs, ilens, ys], subname=subname)

    gen_test(lambda: E2E(idim, odim, args))

    testtools.generate_testcase(lambda: E2E(idim, odim, args, nobias=True),
                           [xs, ilens, ys], backprop=True)


def run(recipe, num_iterations, bwd=True, is_gpu=False):
    (idim, odim, args), (xs, ilens, ys) = recipe
    model = E2E(idim, odim, args, use_chainer=True)

    if is_gpu:
        model.to_gpu()
        xs = chainer.cuda.to_gpu(xs)
        ilens = chainer.cuda.to_gpu(np.array(ilens))
        ys = chainer.cuda.to_gpu(ys)

    cuda_hook = function_hooks.CUDAProfileHook()

    elapsed_total = 0.0
    with cuda_hook:
        for i in range(num_iterations):
            st = time.time()
            model.cleargrads()
            loss = model(xs, ilens, ys)

            if bwd:
                grad = np.ones(loss.shape, loss.dtype)
                if is_gpu:
                    grad = chainer.cuda.to_gpu(grad)
                loss.grad = grad
                loss.backward()

            elapsed = (time.time() - st) * 1000
            print('Elapsed: %s msec' % elapsed)
            if i: elapsed_total += elapsed

    print('Average elapsed: %s msec' % (elapsed_total / (num_iterations - 1)))


def gen(output, recipe, bwd=True, use_gpu=False):
    import testtools
    test_args.get_test_args([output, '--allow-unused-params'])

    (idim, odim, args), (xs, ilens, ys) = recipe
    testtools.generate_testcase(lambda: E2E(idim, odim, args),
                           [xs, ilens, ys], backprop=bwd, use_gpu=use_gpu)


def dispatch():
    parser = argparse.ArgumentParser(description='EspNet E2E')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gen', default=None)
    parser.add_argument('--recipe', default='test', type=str)
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=10)
    args = parser.parse_args()

    recipe = {
        'test': test_recipe,
        'csj': csj_recipe,
        'csj_small': csj_small_recipe,
        'csj_medium': csj_medium_recipe,
        'cpu_bench': cpu_bench_recipe,
        'librispeech': librispeech_recipe,
        'librispeech_small': librispeech_small_recipe,
    }[args.recipe]()

    backward = not args.forward

    if args.gen:
        gen(args.gen, recipe, backward, args.gpu)
    elif args.run:
        run(recipe, args.iterations, backward, is_gpu=args.gpu)
    else:
        raise


def main():
    import numpy as np
    np.random.seed(42)
    gen_test()


if __name__ == '__main__':
    if sys.argv[1].startswith('-'):
        dispatch()
    else:
        main()
