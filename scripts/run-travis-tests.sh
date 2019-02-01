#!/bin/bash

set -eux

bash setup.sh

mkdir build
cd build
cmake .. \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DCHAINERX_BUILD_PYTHON=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python3
make -j2

make large_tests

make test

cd ..
./scripts/runtests.py
pytest python

./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss
