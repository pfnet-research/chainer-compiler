#!/usr/bin/env python3

"""A ResNet50 trainer which is exportable by ONNX-chainer."""

import argparse
import os
import random
import sys

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.functions.evaluation import accuracy
from chainer.training import extensions
import onnx_chainer

import resnet50

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from oniku.tools import npz_to_onnx


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype(np.float32)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def replace_id(model, builtins=__builtins__):
    orig_id = id
    name_map = {}
    param_to_names = {}
    for name, param in model.namedparams():
        param_to_names[id(param)] = name

    def resolve_name(x):
        if orig_id(x) in param_to_names:
            return param_to_names[orig_id(x)]

        param_id = name_map.get(x.name, 0)
        name_map[x.name] = param_id + 1
        name = '%s_%d' % (x.name, param_id) if param_id else x.name
        return name

    def my_id(x):
        if (isinstance(x, chainer.Parameter) or
            isinstance(x, chainer.Variable) and x.name):
            if hasattr(x, 'onnx_name'):
                return x.onnx_name
            name = resolve_name(x)
            setattr(x, 'onnx_name', name)
            return name
        return orig_id(x)
    builtins.id = my_id


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


class MyClassifier(chainer.link.Chain):
    """A Classifier which only supports 2D input."""

    def __init__(self, predictor, compute_accuracy):
        super(MyClassifier, self).__init__()
        self.compute_accuracy = compute_accuracy
        with self.init_scope():
            self.predictor = predictor

    def forward(self, x, t):
        y = self.predictor(x)
        log_softmax = F.log_softmax(y)
        # SelectItem is not supported by onnx-chainer.
        # TODO(hamaji): Support it?
        # log_prob = F.select_item(log_softmax, t)

        # TODO(hamaji): Currently, F.sum with axis=1 cannot be
        # backpropped properly.
        # log_prob = F.sum(log_softmax * t, axis=1)
        # self.batch_size = chainer.Variable(np.array(t.size, np.float32),
        #                                    name='batch_size')
        # return -F.sum(log_prob, axis=0) / self.batch_size
        log_prob = F.sum(log_softmax * t, axis=(0, 1))
        self.batch_size = chainer.Variable(np.array(t.size, np.float32),
                                           name='batch_size')
        loss = -log_prob / self.batch_size
        reporter.report({'loss': loss}, self)
        if self.compute_accuracy:
            acc = accuracy.accuracy(y, np.argmax(t, axis=1))
            reporter.report({'accuracy': acc}, self)
        return loss


class MyIterator(chainer.iterators.MultiprocessIterator):
    """Preprocesses labels to onehot vectors."""

    def __next__(self):
        batch = []
        for input, label in super(MyIterator, self).__next__():
            onehot = np.eye(10, dtype=input.dtype)[label]
            batch.append((input, onehot))
        return batch

    def next(self):
        return self.__next__()


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    main_impl(args)

    # TODO(hamaji): Stop writing a file to scripts.
    with open('scripts/resnet50_stamp', 'w'): pass


def main_impl(args):
    model = resnet50.ResNet50(compute_accuracy=args.run)
    insize = model.insize

    replace_id(model)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Load the datasets and mean file
    mean = np.load(args.mean)

    # Setup an optimizer
    #optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    train = PreprocessedDataset(args.train, args.root, mean, insize)
    val = PreprocessedDataset(args.val, args.root, mean, insize, False)

    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = MyIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = MyIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    if args.run:
        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))
        run_training(args, trainer)
        return

    out_dir = 'out/backprop_test_resnet50'
    makedirs(out_dir)

    # for step in range(2):
    #     trainer.updater.update()
    #     npz_filename = '%s/params_%d.npz' % (out_dir, step)
    #     params_dir = '%s/params_%d' % (out_dir, step)
    #     chainer.serializers.save_npz(npz_filename, model)
    #     makedirs(params_dir)
    #     npz_to_onnx.npz_to_onnx(npz_filename, os.path.join(params_dir, 'param'))

    chainer.config.train = False
    x = np.random.random((args.batchsize, 3, insize, insize)).astype(np.float32)
    y = (np.random.random(args.batchsize) * 1000).astype(np.int32)
    onehot = np.eye(1000, dtype=x.dtype)[y]
    x = chainer.Variable(x, name='input')
    y = chainer.Variable(y, name='y')
    onehot = chainer.Variable(onehot, name='onehot')
    onnx_chainer.export(model, (x, onehot),
                        filename='%s/model.onnx' % out_dir)

    test_data_dir = '%s/test_data_set_0' % out_dir
    makedirs(test_data_dir)
    for i, var in enumerate([x, onehot, model.batch_size]):
        with open(os.path.join(test_data_dir, 'input_%d.pb' % i), 'wb') as f:
            t = npz_to_onnx.np_array_to_onnx(var.name, var.data)
            f.write(t.SerializeToString())

    model.cleargrads()
    result = model(x, onehot)
    result.backward()
    for i, (name, param) in enumerate(model.namedparams()):
        with open(os.path.join(test_data_dir, 'output_%d.pb' % i), 'wb') as f:
            t = npz_to_onnx.np_array_to_onnx('grad_out@' + name, param.grad)
            f.write(t.SerializeToString())


def run_training(args, trainer):
    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

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
