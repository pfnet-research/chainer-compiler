#!/bin/bash

set -eux

cat /proc/cpuinfo
cat /proc/meminfo
nvidia-smi

python3 -m pip install gast
CHAINER_VERSION=$(python3 -c "import imp;print(imp.load_source('_version','third_party/chainer/chainer/_version.py').__version__)")
python3 -m pip install cupy-cuda100==$CHAINER_VERSION

CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 MAKEFLAGS=-j8 \
    CHAINERX_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70 \
    python3 -m pip -q install --no-cache-dir third_party/chainer[test]
python3 -m pip install --no-cache-dir third_party/onnx-chainer[test-gpu]

mkdir -p data
cd data
wget -q https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz && \
    tar -xzf mnist.tar.gz
wget -q https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz && \
    tar -xzf resnet50.tar.gz
cd ..
