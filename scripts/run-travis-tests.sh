#!/bin/bash

set -eux

./scripts/run-clang-format.sh

time sudo pip3 install third_party/chainer
time sudo pip3 install third_party/onnx-chainer

time bash setup.sh

mkdir build
cd build
time cmake .. \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -CHAINER_COMPILER_ENABLE_OPENCV=ON
time make -j2

time make large_tests

time make test

cd ..
./scripts/runtests.py
time pytest -sv python

time python3 examples/mnist/train_mnist.py -d native --compile -I 3 --dump_onnx

time ./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
time ./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss
