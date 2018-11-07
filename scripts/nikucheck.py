#!/usr/bin/python
#
# Example usage:
#
# $ ./scripts/nikucheck.py ch2o/tests/node/Linear.py

import glob
import os
import shutil
import subprocess
import sys


def main():
    if len(sys.argv) == 1:
        sys.stderr.write('Usage: %s test.py\n' % sys.argv[0])
        sys.exit(1)

    py = sys.argv[1]
    tmpdir = 'out/ch2o_tmp'

    shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)
    subprocess.check_call(['python3', py, os.path.join(tmpdir, 'tmp')],
                          env={'PYTHONPATH': 'ch2o'})

    if os.path.exists('build/CMakeCache.txt'):
        build_dir = 'build'
    elif os.path.exists('CMakeCache.txt'):
        build_dir = '.'
    else:
        build_dir = 'build'
    run_onnx = os.path.join(build_dir, 'tools/run_onnx')

    for test in sorted(glob.glob(os.path.join(tmpdir, '*'))):
        print('*** Testing %s ***' % test)
        args = [run_onnx, '--test', test] + sys.argv[2:]
        if 'backprop' in test:
            args.append('--backprop')
        print(' '.join(args))
        subprocess.check_call(args)


main()
