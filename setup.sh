#!/bin/bash
#
# TODO(hamaji): Use cmake and get rid of this shell script.
#

set -e

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

cd ${BASE_DIR}

git submodule update --init

if [ ! -e googletest ]; then
    git clone https://github.com/google/googletest
    (cd googletest && git checkout refs/tags/release-1.8.1)
fi

if [ ! -e googletest/googletest/libgtest.a ]; then
    (cd googletest/googletest && cmake . && make)
fi

if [ ! -e data/mnist/model.onnx ]; then
    rm -rf data/mnist*
    (mkdir -p data && cd data && \
         wget https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz && \
         tar -xvzf mnist.tar.gz)
fi

if [ ! -e data/resnet50/model.onnx ]; then
    rm -rf data/resnet50*
    (mkdir -p data && cd data && \
         wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz && \
         tar -xvzf resnet50.tar.gz)
fi

if [ ! -e gsl-lite/include/gsl/gsl ]; then
    git clone https://github.com/martinmoene/gsl-lite
fi

if [ ! -e optional-lite/include/nonstd/optional.hpp ]; then
    git clone https://github.com/martinmoene/optional-lite
fi
