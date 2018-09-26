#!/bin/bash
#
# Example usage:
#
# $ ./scripts/nikucheck.sh ch2o/tests/ListComp.py

set -e

rm -fr out/ch2o_tmp
mkdir -p out/ch2o_tmp
PYTHONPATH=ch2o python3 "$@" out/ch2o_tmp/tmp

for i in out/ch2o_tmp/*; do
    echo "*** Testing $i ***"
    ./tools/run_onnx --test $i
done
