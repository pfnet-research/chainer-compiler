#!/bin/bash

set -euc

sudo apt-get install -y --no-install-recommends \
     libprotobuf-dev protobuf-compiler libopencv-dev

bash setup.sh
pip install gast numpy chainer onnx==1.3.0 onnx_chainer

mkdir build
cd build
cmake .. -DCHAINER_COMPILER_BUILD_CUDA=OFF -DCHAINER_COMPILER_ENABLE_PYTHON=ON
make -j2

make test
./scripts/runtests.py
pytest python
