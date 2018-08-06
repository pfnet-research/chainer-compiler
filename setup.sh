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
