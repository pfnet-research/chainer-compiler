#!/usr/bin/env python

"""An MNIST trainer which is exportable by ONNX-chainer."""

import argparse
import os
import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import onnx_chainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oniku.tools import npz_to_onnx


class MyClassifier(chainer.link.Chain):
    """A Classifier which only supports 2D input."""

    def __init__(self, predictor):
        super(MyClassifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def forward(self, x, t):
        y = self.predictor(x)
        log_softmax = F.log_softmax(y)
        # SelectItem is not supported by onnx-chainer.
        # TODO(hamaji): Support it?
        # log_prob = F.select_item(log_softmax, t)
        log_prob = F.sum(log_softmax * t, axis=1)
        batch_size = chainer.Variable(np.array(t.size, np.float32))
        return -F.sum(log_prob, axis=0) / batch_size


class MyIterator(chainer.iterators.SerialIterator):
    """Preprocesses labels to onehot vectors."""

    def __next__(self):
        batch = []
        for input, label in super(MyIterator, self).__next__():
            onehot = np.eye(10, dtype=input.dtype)[label]
            batch.append((input, onehot))
        return batch

    def next(self):
        return self.__next__()


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out, use_sigmoid=False):
        super(MLP, self).__init__()
        self.activation_fn = self.sigmoid if use_sigmoid else F.relu
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def sigmoid(self, x):
        a = chainer.Variable(np.ones(x.shape, np.float32))
        return a / (a + F.exp(-x))

    def __call__(self, x):
        h1 = self.activation_fn(self.l1(x))
        h2 = self.activation_fn(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=7,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--onnx', default='',
                        help='Export ONNX model')
    parser.add_argument('--model', '-m', default='model.npz',
                        help='Model file name to serialize')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Enable timeout')
    parser.add_argument('--trace', default='',
                        help='Enable tracing')
    args = parser.parse_args()

    main_impl(args)


def main_impl(args):
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = MLP(args.unit, 10, use_sigmoid=True)
    # classifier = L.Classifier(model)
    classifier = MyClassifier(model)

    model = classifier

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    #optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = MyIterator(train, args.batchsize, shuffle=False)
    test_iter = MyIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    out_dir = 'out/mnist_mlp'
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    for step in range(2):
        trainer.updater.update()
        npz_filename = '%s/params_%d.npz' % (out_dir, step)
        params_dir = '%s/params_%d' % (out_dir, step)
        chainer.serializers.save_npz(npz_filename, model)
        if not os.path.exists(params_dir): os.makedirs(params_dir)
        npz_to_onnx.npz_to_onnx(npz_filename, os.path.join(params_dir, 'param'))

    chainer.config.train = False
    x = np.zeros((args.batchsize, 784), dtype=np.float32)
    # y = np.zeros((1), dtype=np.int32)
    y = np.zeros((args.batchsize, 10), dtype=np.float32)
    onnx_chainer.export(model, (x, y), filename='%s/model.onnx' % out_dir)


if __name__ == '__main__':
    main()
