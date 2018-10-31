#!/bin/bash

set -e

rm -fr onnx/build
mkdir -p onnx/build
cd onnx/build
cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DONNX_ML=ON \
    ..
cmake --build . -- -j$(nproc)
