#!/bin/bash

set -eux

./scripts/run-clang-format.sh

sudo pip3 install third_party/chainer
sudo pip3 install third_party/onnx-chainer

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
pytest -sv python

python3 examples/mnist/train_mnist.py -d native --compile -I 3 --dump_onnx

# TODO(hamaji): Investigate why running these on Travis takes ~10 minutes.
# ./build/tools/run_onnx --test out/ch2o_model_Alex_with_loss
# ./build/tools/run_onnx --test out/ch2o_model_GoogleNet_with_loss
