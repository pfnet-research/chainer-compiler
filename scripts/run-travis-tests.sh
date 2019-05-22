#!/bin/bash

set -eux

./scripts/run-clang-format.sh

time sudo pip3 install third_party/chainer
time sudo pip3 install third_party/onnx-chainer
# TODO(hamaji): Remove this once ONNX-chainer becomes compatible with
# ONNX-1.5.0.
time sudo pip3 install onnx==1.4.1

export LD_LIBRARY_PATH="$HOME/.cudnn/active/cuda/lib64:/usr/local/cuda-10.0/lib64/stubs:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.cudnn/active/cuda/include:${CPATH:-}"
export LIBRARY_PATH="$HOME/.cudnn/active/cuda/lib64:${LIBRARY_PATH:-}"

time bash setup.sh

mkdir build
cd build
time cmake .. \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DCHAINER_COMPILER_ENABLE_CUDA=ON \
      -DCHAINER_COMPILER_ENABLE_CUDNN=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0 \
      -DCUDNN_ROOT_DIR=$HOME/.cudnn/active/cuda \
      -DCHAINER_COMPILER_USE_CUDA_STUBS=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DCHAINER_COMPILER_ENABLE_OPENCV=ON
time make -j2

time make large_tests

if time make test ; then
    exit_code=$?
    cat ./Testing/Temporary/LastTest.log
    exit $?
fi

cd ..
./scripts/runtests.py
time pytest -sv python

time python3 examples/mnist/train_mnist.py \
     -d native --compile -I 3 --use-fake-data

time ./build/tools/dump out/ch2o_model_MLP_with_loss

time ./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
time ./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss
