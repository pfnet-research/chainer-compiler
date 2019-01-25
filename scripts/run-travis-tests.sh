#!/bin/bash

set -eux

bash setup.sh

mkdir build
cd build
cmake .. \
      -DCHAINER_COMPILER_BUILD_CUDA=OFF \
      -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
      -DCMAKE_C_COMPILER=/usr/lib/ccache/gcc \
      -DCMAKE_C_COMPILER=/usr/lib/ccache/g++
make -j2

make test

cd ..
./scripts/runtests.py
pytest python
