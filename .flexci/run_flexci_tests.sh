#!/bin/bash

set -eux

cat <<'EOM' >runtest.sh
set -eux

. .flexci/before_install.sh

mkdir -p build
cd build
cmake .. \
    -DCHAINER_COMPILER_ENABLE_CUDA=ON \
    -DCHAINER_COMPILER_ENABLE_CUDNN=ON \
    -DCHAINER_COMPILER_ENABLE_OPENCV=ON \
    -DCHAINER_COMPILER_ENABLE_PYTHON=ON \
    -DCHAINER_COMPILER_ENABLE_TVM=ON \
    -DCHAINER_COMPILER_ENABLE_TENSORRT=ON \
    -DCHAINER_COMPILER_TVM_DIR=$HOME/tvm_dist \
    -DCHAINER_COMPILER_TVM_INCLUDE_DIRS="/root/tvm_dist/include;/root/dmlc_core_dist/include;/root/dlpack/include;/root/tvm_dist/include/HalideIR" \
    -DCHAINER_COMPILER_NGRAPH_DIR=$HOME/ngraph_dist \
    -DCHAINER_COMPILER_DLDT_DIR=$HOME/dldt \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCHAINERX_BUILD_CUDA=ON \
    -DCHAINERX_BUILD_PYTHON=ON \
    -DCHAINER_COMPILER_PREBUILT_CHAINERX_DIR=$(pip3 show chainer | awk '/^Location: / {print $2}')/chainerx \
    && \
    make -j8

make large_tests

ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
make test
unlink /usr/local/cuda/lib64/stubs/libcuda.so.1
cd ..

./build/tools/run_onnx data/shufflenet --compiler_log --use_dldt
./build/tools/run_onnx data/shufflenet --compiler_log -d cuda --use_tensorrt

PYTHONPATH=. python3 scripts/runtests.py -g --fuse --target_opsets=8,9,10
LD_LIBRARY_PATH=$HOME/ngraph_dist/lib:$LD_LIBRARY_PATH PYTHONPATH=. python3 scripts/runtests.py --ngraph
PYTHONPATH=. python3 -m pytest -sv tests

EOM

. .flexci/common.sh

. .flexci/cache.sh
pull_chainer_whl

docker run --runtime=nvidia --memory-swap=-1 --rm -v=$(pwd):/chainer-compiler --workdir=/chainer-compiler \
    ${CI_IMAGE} /bin/bash /chainer-compiler/runtest.sh

push_chainer_whl
