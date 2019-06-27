#!/usr/bin/env python
import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

import chainer_compiler


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def fake_dataset():
    def gen(size):
        inputs = []
        labels = []
        for i in range(size):
            inputs.append(np.random.rand(784).astype(np.float32))
            labels.append(np.random.randint(10))
        return inputs, labels
    train = chainer.datasets.TupleDataset(*gen(150))
    test = chainer.datasets.TupleDataset(*gen(100))
    return train, test


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--export', type=str, default=None,
                        help='Export the model to ONNX')
    parser.add_argument('--compile', type=str, default=None,
                        help='Compile the model')

    parser.add_argument('--dump_onnx', action='store_true',
                        help='Dump ONNX model after optimization')
    parser.add_argument('--iterations', '-I', type=int, default=None,
                        help='Number of iterations to train')
    parser.add_argument('--use-fake-data', action='store_true',
                        help='Use fake data')
    parser.add_argument('--translator', type=str, default='onnx_chainer',
                        help='ch2o or onnx_chainer')
    parser.add_argument('--computation_order', type=str, default=None,
                        help='Computation order in backpropagation')
    parser.add_argument('--use_unified_memory', dest='use_unified_memory',
                        action='store_true',
                        help='Use unified memory for large model')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    mlp = MLP(args.unit, 10)

    if args.export is not None:
        if args.use_unified_memory:
            chainer_compiler.use_unified_memory_allocator()
        mlp.to_device(device)
        x = mlp.xp.zeros((args.batchsize, 784)).astype(np.float32)
        chainer_compiler.export(mlp, [x], args.export, args.translator)
        return

    if args.compile is not None:
        chainer_compiler.use_chainerx_shared_allocator()
        mlp.to_device(device)
        with chainer.using_config('enable_backprop', False),\
                chainer.using_config('train', False):
            x = mlp.xp.zeros((1, 784)).astype(np.float32)
            mlp(x)  # initialize model parameters before compile
        mlp = chainer_compiler.compile_onnx(
            mlp,
            args.compile,
            args.translator,
            dump_onnx=args.dump_onnx,
            computation_order=args.computation_order)

    model = L.Classifier(mlp)
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    if args.use_fake_data:
        train, test = fake_dataset()
    else:
        train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    if args.iterations:
        stop_trigger = (args.iterations, 'iteration')
    else:
        stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # TODO(niboshi): Temporarily disabled for chainerx. Fix it.
    if device.xp is not chainerx:
        trainer.extend(extensions.DumpGraph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
