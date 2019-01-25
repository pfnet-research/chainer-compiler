#!/bin/bash

set -eux

bash setup.sh

mkdir build
cd build
cmake .. \
      -DCHAINER_COMPILER_BUILD_CUDA=OFF \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON
make -j2

make test

cd ..
./scripts/runtests.py
# TODO(hamaji): Enable Python test.
# https://github.com/pfnet-research/chainer-compiler/issues/2
# pytest python
