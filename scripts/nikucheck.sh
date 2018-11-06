#!/bin/bash
#
# Example usage:
#
# $ ./scripts/nikucheck.sh ch2o/tests/node/Linear.py

set -e

py=$1
shift

rm -fr out/ch2o_tmp
mkdir -p out/ch2o_tmp
PYTHONPATH=ch2o python3 "${py}" out/ch2o_tmp/tmp

if [ -e build/CMakeCache.txt ]; then
    run_onnx=./build/tools/run_onnx
elif [ -e CMakeCache.txt ]; then
    run_onnx=./tools/run_onnx
else
    run_onnx=./build/tools/run_onnx
fi

for i in out/ch2o_tmp/*; do
    echo "*** Testing $i ***"
    args="$@"
    if echo $i | grep backprop; then
        args+=--backprop
    fi
    echo "${run_onnx}" --test $i "${args}"
    "${run_onnx}" --test $i "${args}"
done
