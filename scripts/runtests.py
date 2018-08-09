#!/usr/bin/python3

import os
import sys
import subprocess


class TestCase(object):

    def __init__(self, dirname, name):
        self.dirname = dirname
        self.name = name

    def test_dir(self):
        return os.path.join(self.dirname, self.name)


NODE_TEST = 'onnx/onnx/backend/test/data/node'


TEST_CASES = [
    TestCase(NODE_TEST, 'test_add'),
    TestCase(NODE_TEST, 'test_add_bcast'),
    TestCase(NODE_TEST, 'test_relu'),
]


def main():
    if os.path.exists('Makefile'):
        subprocess.check_call('make -j4')
    elif os.path.exists('build.ninja'):
        subprocess.check_call('ninja')

    fail_cnt = 0
    for test_case in TEST_CASES:
        sys.stdout.write('%s... ' % test_case.name)
        args = ['tools/run_onnx', '--test', test_case.test_dir(), '--quiet', '1']
        try:
            subprocess.check_call(args)
            sys.stdout.write('OK\n')
        except subprocess.CalledProcessError:
            fail_cnt += 1
            sys.stdout.write('FAIL\n')
    if fail_cnt:
        print(f'{fail_cnt} tests failed!')
    else:
        print('ALL OK!')


main()
