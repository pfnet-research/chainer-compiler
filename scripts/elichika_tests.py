#!/usr/bin/python

import importlib
import glob
import os
import subprocess
import sys

from test_case import TestCase


class Generator(object):
    def __init__(self, dirname, filename):
        self.dirname = dirname
        self.filename = filename


# TODO(hamaji): Triage failing tests.
TESTS = [
    # Generator('node', 'Convolution2D'),
    Generator('node', 'Linear'),
    # Generator('node', 'Relu'),
    # Generator('node', 'Softmax'),

    # Generator('syntax', 'ChinerFunctionNode'),
    # Generator('syntax', 'Cmp'),
    # Generator('syntax', 'For'),
    # Generator('syntax', 'ForAndIf'),
    # Generator('syntax', 'If'),
    # Generator('syntax', 'LinkInFor'),
    # Generator('syntax', 'ListComp'),
    Generator('syntax', 'MultiClass'),
    Generator('syntax', 'MultiFunction'),
    # Generator('syntax', 'Range'),
    # Generator('syntax', 'Sequence'),
    # Generator('syntax', 'Slice'),
    # Generator('syntax', 'UserDefinedFunc'),
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

    # Force re-run cmake as the dependency must be updated when the
    # list of test names changes.
    # TODO(hamaji): Come up with a better way to tell CMake the need
    # of re-generation.
    myname = sys.argv[0]
    cmake_list = os.path.join(get_source_dir(), 'CMakeLists.txt')
    if os.stat(cmake_list).st_mtime < os.stat(myname).st_mtime:
        os.utime(cmake_list)

    for gen in get_test_generators(dirname):
        py = os.path.join('tests', gen.dirname, gen.filename)
        out_dir = os.path.join(get_source_dir(), 'out', 'elichika_%s_%s' %
                               (gen.dirname, gen.filename))
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
        category = gen.dirname
        name = gen.filename
        test_name = 'elichika_%s_%s' % (category, name)
        kwargs = {}

        diversed = False
        for substr in diversed_whitelist:
            if substr in test_name:
                diversed = True
                break

        test_dirs = glob.glob('out/%s' % test_name)
        test_dirs += glob.glob('out/%s_*' % test_name)
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
