#!/bin/bash

set -eux

bash setup.sh

mkdir build
cd build
cmake .. \
      -DCHAINER_COMPILER_BUILD_CUDA=OFF \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DCHAINERX_BUILD_PYTHON=ON
make -j2

make large_tests

make test

cd ..
./scripts/runtests.py
# TODO(hamaji): Enable Python test.
# https://github.com/pfnet-research/chainer-compiler/issues/2
# pytest python

./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss

ABC 01
BAC 12
BCA 02
ACB 12
ABC
