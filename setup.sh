#!/bin/bash
#
# TODO(hamaji): Use cmake and get rid of this shell script.
#

set -e

git submodule update --init

if [ ! -e onnx/build/libonnx.a -o ! -e onnx/build/onnx/onnx-ml.pb.h ]; then
    (cd onnx && \
        ([ -d "build" ] || mkdir -p build) && \
        cd build && \
        cmake \
            -DBUILD_ONNX_PYTHON=OFF \
            -DCMAKE_CXX_FLAGS=-fPIC \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DONNX_ML=1 \
            -DONNX_NAMESPACE=onnx \
            .. && \
        cmake --build . -- -j$(nproc))
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

# CMake would be confused when there are no generated code yet.
# TODO(hamaji): Remove this by fixing dependency specified in
# runtime/CMakeLists.txt.
(cd runtime && python3 gen_xcvm.py)
