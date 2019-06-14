#!/bin/bash

set -eux

./scripts/run-clang-format.sh

run() {
    set +x
    travis_fold start $1
    travis_time_start
    local n=$1
    shift
    echo "Command: $@"
    /usr/bin/time "$@"
    travis_time_finish
    set -x
    travis_fold end $n
}

run pip_chainer sudo pip3 install third_party/chainer
run pip_onnx_chainer sudo pip3 install third_party/onnx-chainer onnxruntime==0.4.0 onnx==1.5.0

run pip_list pip3 list -v

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
PYTHONPATH=. run runtests ./scripts/runtests.py 2>&1
PYTHONPATH=. run pytest pytest -sv tests
PYTHONPATH=. run canonicalizer_tests pytest testcases/elichika_tests/canonicalizer

PYTHONPATH=. run train_mnist python3 examples/mnist/train_mnist.py \
     -d native --compile -I 3 --use-fake-data

run tools_dump ./build/tools/dump out/ch2o_model_MLP_with_loss

run run_onnx_verbose \
    ./build/tools/run_onnx --test out/ch2o_model_MLP_with_loss \
    --verbose --compiler_log
run run_onnx_trace sh -c \
    './build/tools/run_onnx --test out/ch2o_model_EspNet_E2E --trace 2>&1 | head -100'

run run_onnx_alex \
    ./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
run run_onnx_googlenet \
    ./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss
