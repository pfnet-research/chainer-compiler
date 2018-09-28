#!/usr/bin/python3
#
# Usage:
#
# $ ./scripts/runtests.py -g onnx_real --show_log |& tee log
# $ ./scripts/parse_elapsed.py log

import re
import sys

tests = {}

cur_test = None
with open(sys.argv[1]) as f:
    for line in f:
        m = re.match(r'^Running for out/onnx_real_(.*?)/', line)
        if m:
            cur_test = m.group(1)
            continue
        m = re.match(r'^Elapsed: (\d+\.\d+)', line)
        if m:
            tests[cur_test] = float(m.group(1))

for name, elapsed in sorted(tests.items()):
    print('%s %s' % (name, elapsed))
