#!/bin/bash
#
# TODO(hamaji): Use cmake and get rid of this shell script.
#

set -e

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

cd ${BASE_DIR}

git submodule update --init

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
