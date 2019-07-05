# Compare two output dumps created by --dump_outputs_dir.
#
# $ mkdir a b
# $ build/tools/run_onnx --dump_outputs_dir a  --backprop --test out/backprop_test_mnist_mlp
# ... do some change ...
# $ build/tools/run_onnx --dump_outputs_dir b  --backprop --test out/backprop_test_mnist_mlp
# $ python3 tools/compare_dump_dirs.py a b
#

import argparse
import glob
import os
import re
import sys

import numpy as np


def read_dump_dir(d):
    filenames = sorted(glob.glob(os.path.join(d, '*.npy')))
    files = []
    for filename in filenames:
        name = os.path.basename(filename)
        matched = re.match(r'\d+_(.*)\.npy$', name)
        if matched:
            files.append((matched.group(1), filename))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--atol', type=float, default=0)
    args = parser.parse_args()


    files1 = read_dump_dir(args.dir1)
    files2 = read_dump_dir(args.dir2)

    files_map2 = dict(files2)

    for name, filename1 in files1:
        filename2 = files_map2.get(name)
        if filename2 is None:
            continue
        sys.stderr.write('Comparing %s...' % name)
        n1 = np.load(filename1)
        n2 = np.load(filename2)
        try:
            np.testing.assert_allclose(n1, n2, rtol=args.rtol, atol=args.atol)
            sys.stderr.write(' OK\n')
        except AssertionError as e:
            sys.stderr.write('%s\n\n' % e)
            sys.stderr.write('=== %s ===\n%s\n' % (filename1, n1))
            sys.stderr.write('=== %s ===\n%s\n' % (filename2, n2))


if __name__ == '__main__':
    main()
