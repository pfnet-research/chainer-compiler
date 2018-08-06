#!/bin/bash
#
# TODO(hamaji): Use cmake and get rid of this shell script.
#

if [ ! -e onnx ]; then
    git clone https://github.com/onnx/onnx
fi

if [ ! -e onnx/.setuptools-cmake-build/libonnx.a ]; then
    (cd onnx && python3 setup.py build)
fi

if [ ! -e googletest ]; then
    git clone https://github.com/google/googletest
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

if [ ! -e gsl-lite/include/gsl/gsl ]; then
    git clone https://github.com/martinmoene/gsl-lite
fi

if [ ! -e optional-lite/include/nonstd/optional.hpp ]; then
    git clone https://github.com/martinmoene/optional-lite
fi
