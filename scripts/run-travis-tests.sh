#!/bin/bash

set -euc

bash setup.sh

mkdir build
cd build
cmake .. -DCHAINER_COMPILER_BUILD_CUDA=OFF -DCHAINER_COMPILER_ENABLE_PYTHON=ON
make -j2

make test
./scripts/runtests.py
pytest python
