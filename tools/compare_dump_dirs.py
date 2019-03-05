# Compare two output dumps created by --dump_output_dir.

import glob
import os
import re
import sys

import numpy as np


if len(sys.argv) != 3:
    sys.stderr.write('Usage: %s <dir1> <dir2>\n' % sys.argv[0])
    sys.exit(1)


def read_dump_dir(d):
    filenames = sorted(glob.glob(os.path.join(d, '*.npy')))
    files = []
    for filename in filenames:
        name = os.path.basename(filename)
        matched = re.match(r'\d+_(.*)\.npy$', name)
        if matched:
            files.append((matched.group(1), filename))
    return files


files1 = read_dump_dir(sys.argv[1])
files2 = read_dump_dir(sys.argv[2])

files_map2 = dict(files2)

for name, filename1 in files1:
    filename2 = files_map2.get(name)
    if filename2 is None:
        continue
    print('Comparing %s' % name)
    n1 = np.load(filename1)
    n2 = np.load(filename2)
    try:
        np.testing.assert_allclose(n1, n2)
    except AssertionError as e:
        sys.stderr.write('%s\n\n' % e)
        sys.stderr.write('=== %s ===\n%s\n' % (filename1, n1))
        sys.stderr.write('=== %s ===\n%s\n' % (filename2, n2))
