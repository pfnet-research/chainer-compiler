#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

cat /proc/cpuinfo
cat /proc/meminfo
nvidia-smi

python3 -m pip install gast
python3 -m pip install --pre cupy-cuda100==7.0.0a1

CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS=-j8 \
    CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
    python3 -m pip -q install --no-cache-dir third_party/chainer[test]
python3 -m pip install --no-cache-dir third_party/onnx-chainer[test-gpu]

mkdir -p data
cd data
wget https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz && \
    tar -xzf mnist.tar.gz
wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz && \
    tar -xzf resnet50.tar.gz
cd ..

mkdir -p build
cd build
cmake .. \
    -DCHAINER_COMPILER_ENABLE_CUDA=ON \
    -DCHAINER_COMPILER_ENABLE_CUDNN=ON \
    -DCHAINER_COMPILER_ENABLE_OPENCV=ON \
    -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
    -DCHAINER_COMPILER_NGRAPH_DIR=$HOME/ngraph_dist \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCHAINER_COMPILER_ENABLE_TVM=ON \
    -DCHAINERX_BUILD_CUDA=ON \
    -DCHAINERX_BUILD_PYTHON=ON \
    -DCHAINER_COMPILER_PREBUILT_CHAINERX_DIR=$(pip3 show chainer | awk '/^Location: / {print $2}')/chainerx \
    && \
    make -j8

make large_tests

make test
cd ..

PYTHONPATH=. python3 scripts/runtests.py -g --fuse
PYTHONPATH=. python3 scripts/runtests.py --ngraph
PYTHONPATH=. python3 -m pytest -sv tests

EOM

docker run --runtime=nvidia --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    disktnk/chainer-compiler:ci-cuda100-ngraph0.19.0-tvm0.5 /bin/bash /chainer-compiler/runtest.sh
