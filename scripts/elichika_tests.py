#!/usr/bin/python

# This file is included by CMakeLists.txt.
#[[

import importlib
import glob
import os
import subprocess
import sys

from test_case import TestCase


class Generator(object):
    def __init__(self, dirname, filename, fail=False):
        self.dirname = dirname
        self.category = dirname.replace('/', '_')
        self.filename = filename
        self.fail = True


TESTS = [
    # TODO(hamaji): Remove `fail=True` after fixing the implementation
    # of SoftmaxCrossEntropy.
    Generator('model', 'MLP', fail=True),
    # TODO(hamaji): Build and run this test.
    # Generator('model', 'Alex'),

    Generator('node', 'AddMul'),
    Generator('node', 'AveragePool2d'),
    Generator('node', 'BatchNorm'),
    Generator('node', 'Convolution2D'),
    Generator('node', 'Id'),
    Generator('node', 'Linear'),
    Generator('node', 'PadSequence'),
    Generator('node', 'Relu'),
    Generator('node', 'Softmax'),
    Generator('node', 'SoftmaxCrossEntropy'),
    Generator('node', 'Unpooling2D'),
    Generator('node', 'Variable'),

    Generator('node/ndarray', 'NpArray'),
    Generator('node/ndarray', 'NpFull'),
    Generator('node/ndarray', 'NpZeros'),
    Generator('node/ndarray', 'Size'),
    Generator('node/ndarray', 'Shape'),
    Generator('node/ndarray', 'Ceil'),

    Generator('node/Functions', 'Reshape'),
    Generator('node/Functions', 'SplitAxis'),
    Generator('node/Functions', 'Roi'),
    Generator('node/Functions', 'SwapAxes'),
    Generator('node/Functions', 'Concat'),
    Generator('node/Functions', 'Dropout'),
    Generator('node/Functions', 'Matmul'),
    Generator('node/Functions', 'MaxPool2d'),
    Generator('node/Functions', 'ResizeImages'),

    Generator('syntax', 'Alias'),
    Generator('syntax', 'Cmp'),
    Generator('syntax', 'For'),
    Generator('syntax', 'ForAndIf'),
    Generator('syntax', 'If'),
    Generator('syntax', 'LinkInFor'),
    Generator('syntax', 'ListComp'),
    Generator('syntax', 'MultiClass'),
    Generator('syntax', 'MultiFunction'),
    Generator('syntax', 'Range'),
    Generator('syntax', 'Sequence'),
    Generator('syntax', 'Slice'),
    Generator('syntax', 'UserDefinedFunc'),
    Generator('syntax', 'Tuple'),
]


def get_test_generators(dirname):
    return [test for test in TESTS if test.dirname == dirname]


def print_test_generators(dirname):
    tests = []
    for gen in get_test_generators(dirname):
        tests.append(
            os.path.join('elichika/tests', gen.dirname, gen.filename + '.py'))
    print(';'.join(tests))


def get_source_dir():
    return os.path.dirname(os.path.dirname(sys.argv[0]))


def generate_tests(dirname):
    from testtools import testcasegen

    for gen in get_test_generators(dirname):
        py = os.path.join('tests', gen.dirname, gen.filename)
        out_dir = os.path.join(get_source_dir(), 'out', 'elichika_%s_%s' %
                               (gen.category, gen.filename))
        print('Running %s' % py)
        module = importlib.import_module(py.replace('/', '.'))
        testcasegen.reset_test_generator([out_dir])
        module.main()


def get():
    tests = []

    diversed_whitelist = [
        'node_Linear'
    ]

    for gen in TESTS:
        category = gen.category
        name = gen.filename
        test_name = 'elichika_%s_%s' % (category, name)
        kwargs = {}

        if gen.fail:
            kwargs['fail'] = True

        diversed = False
        for substr in diversed_whitelist:
            if substr in test_name:
                diversed = True
                break

        test_dirs = glob.glob('out/%s' % test_name)
        test_dirs += glob.glob('out/%s_*' % test_name)
        assert test_dirs, 'No tests found for %s' % test_name
        for d in test_dirs:
            name = os.path.basename(d)
            test_dir = os.path.join('out', name)
            tests.append(TestCase(name=name, test_dir=test_dir, **kwargs))
            if diversed:
                tests.append(TestCase(name=name + '_diversed',
                                      test_dir=test_dir,
                                      backend='xcvm_test',
                                      **kwargs))

    return tests


if __name__ == '__main__':
    if sys.argv[1] == '--list':
        print_test_generators(sys.argv[2])
    elif sys.argv[1] == '--generate':
        generate_tests(sys.argv[2])
    else:
        raise RuntimeError('See %s for the usage' % sys.argv[0])

#]]
