#!/bin/bash

# TODO(hamaji): Revive -u if possible.
# set -eux
set -ex

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

pushd third_party/chainer
CHAINER_WHL_CACHE_DIR=$HOME/dist-chainer/$TRAVIS_OS_NAME/$(git rev-parse --short HEAD)
popd
if [[ -d $CHAINER_WHL_CACHE_DIR ]]; then
  echo "Use cached chainer wheel"
else
  run pip_wheel sudo pip3 install wheel
  pushd third_party/chainer
  run chainer_whl python3 setup.py bdist_wheel
  popd
  rm -rf $HOME/dist-chainer/*
  mkdir -p $CHAINER_WHL_CACHE_DIR
  cp -p third_party/chainer/dist/*.whl $CHAINER_WHL_CACHE_DIR
fi
run pip_chainer sudo pip3 install $CHAINER_WHL_CACHE_DIR/*.whl
run pip_dependencies sudo pip3 install --no-color gast==0.3.2 chainercv 'onnx>=1.4.0,<1.6' 'pytest<5.0.0' zipp==1.0.0 torch==1.4.0

run pip_list pip3 list -v

mkdir build
cd build
run cmake cmake .. \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DCHAINER_COMPILER_ENABLE_OPENCV=ON \
      -DCHAINER_COMPILER_PREBUILT_CHAINERX_DIR=$(pip3 show chainer | awk '/^Location: / {print $2}')/chainerx
run make make -j2

run large_tests make large_tests

run unit_tests ctest -V

cd ..
./scripts/checkout-onnx-worktrees.sh
PYTHONPATH=. run runtests ./scripts/runtests.py --target_opsets=8,9,10 2>&1
PYTHONPATH=. run pytest pytest -sv tests
PYTHONPATH=. run canonicalizer_tests pytest testcases/elichika_tests/canonicalizer

PYTHONPATH=. run train_mnist_export python3 examples/mnist/train_mnist.py \
     -d -1 --export /tmp/tmp_mnist_model.onnx -I 3 --use-fake-data
PYTHONPATH=. run train_mnist_compile python3 examples/mnist/train_mnist.py \
     -d native --compile /tmp/tmp_mnist_model.onnx -I 3 --use-fake-data

mkdir -p npy_outputs
run dump_outputs_dir ./build/tools/run_onnx out/elichika_syntax_For_basic1 \
    --dump_outputs_dir=npy_outputs
# There should be at least a single output dump.
ls -l npy_outputs/*.npy

run tools_dump ./build/tools/dump out/elichika_model_MLP_backprop

run run_onnx_verbose \
    ./build/tools/run_onnx --test out/elichika_model_MLP \
    --verbose --compiler_log --chrome_tracing mlp.json
ls -l mlp.json

run run_onnx_trace sh -c \
    './build/tools/run_onnx --test out/elichika_model_EspNet_E2E --trace 2>&1 | head -100'

run run_onnx_alex \
    ./build/tools/run_onnx out/elichika_model_Alex \
    --check_infs --check_nans --strip_chxvm
run run_onnx_googlenet \
    ./build/tools/run_onnx out/elichika_model_GoogleNet
