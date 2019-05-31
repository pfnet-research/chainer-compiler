#!/bin/bash

set -eux

./scripts/run-clang-format.sh

run() {
    local n=$1
    shift
    echo travis_fold:start:$n
    /usr/bin/time "$@"
    echo travis_fold:end:$n
}

run pip_chainer sudo pip3 install third_party/chainer
run pip_onnx_chainer sudo pip3 install third_party/onnx-chainer
# TODO(hamaji): Remove this once ONNX-chainer becomes compatible with
# ONNX-1.5.0.
run pip_onnx sudo pip3 install onnx==1.4.1

run setup_sh bash setup.sh

mkdir build
cd build
run cmake cmake .. \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DCHAINER_COMPILER_ENABLE_OPENCV=ON \
      -DCHAINER_COMPILER_PREBUILT_CHAINERX_DIR=$(pip3 show chainer | awk '/^Location: / {print $2}')/chainerx
run make make -j2

run large_tests make large_tests

run unit_tests make test

cd ..
./scripts/runtests.py
run pytest pytest -sv python

run train_mnist python3 examples/mnist/train_mnist.py \
     -d native --compile -I 3 --use-fake-data

run tools_dump ./build/tools/dump out/ch2o_model_MLP_with_loss

run run_onnx_verbose \
    ./build/tools/run_onnx --test out/ch2o_model_MLP_with_loss --verbose
run run_onnx_trace \
    ./build/tools/run_onnx --test out/ch2o_model_EspNet_E2E --trace

run run_onnx_alex \
    ./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
run run_onnx_googlenet \
    ./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss
