#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

. .chainerci/before_install.sh

mkdir -p build
cd build
cmake .. \
    -DCHAINER_COMPILER_ENABLE_CUDA=ON \
    -DCHAINER_COMPILER_ENABLE_CUDNN=ON \
    -DCHAINER_COMPILER_ENABLE_OPENCV=ON \
    -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
    -DCHAINER_COMPILER_NGRAPH_DIR=$HOME/ngraph_dist \
    -DCHAINER_COMPILER_DLDT_DIR=$HOME/dldt \
    -DPYTHON_EXECUTABLE=$(which python3) \
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
    disktnk/chainer-compiler:ci-base-7c293fc /bin/bash /chainer-compiler/runtest.sh
